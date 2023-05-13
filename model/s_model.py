import codecs
from embedding.bert_embedding import Bert_embedding
from os.path import dirname, abspath
import sys
import numpy as np
import torch
import torch.nn as nn
from numpy.random import randint
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')
from utils.function import *
from utils.logger import Logger
from embedding.torchEmbedding import TorchEmbedding, PositionEmbedding, EmbeddingProcessor
from embedding.AlbertEmbedding import AlbertEmbedding
from model.attention import SelfAttention, MultiHeadAttention
from model.loss_function import DynamicFocalLossBCE, FocalLossBCE
from model.Decoder import DGCNNDecoder
from model.Encoder import RNNEncoder
from model.model_utils import EMA
from config import Config

log = Logger('s_model', std_out=False, save2file=True).get_logger()

parent_path = abspath(dirname(dirname(__file__)))
TRAIN_FILE = parent_path + '/data/my_train_data.json'
TEST_FILE = parent_path + '/data/my_test_data.json'
ID_PREDICATE_FILE = parent_path + '/data/id_and_predicate_no_unk.json'
PREDICATE_INFO_FILE = parent_path + '/data/predicate_info.json'
CHAR2ID_FILE = parent_path + '/data/char2id.json'

class TextData(object):
    '''
    加载文本文件到内存，每条数据对应一个输入、一个输出
    json_file：json文件的绝对路径
    id_predicate_file: id_predicate_file
    '''
    def __init__(self, json_file: str, id_predicate_file: str=ID_PREDICATE_FILE, char2id_file: str=CHAR2ID_FILE):
        '''
        json_file：json文件的绝对路径
        '''
        super().__init__()
        raw_data = read_json(json_file)
        
       
        with codecs.open(char2id_file, 'r', encoding='utf-8') as f:
            self.char2id = ujson.load(f)

        self.max_seq_len, self.subject_max_len = self.__compute_max_len(raw_data)
        self.len = len(raw_data)
        
        self.inputs_outputs = self.__process_data_for_dataloader(raw_data)

    def __process_data_for_dataloader(self, raw_data: list):
        '''
        '''
        text = []
        spo_list = []
        choice_index = []

        print('process data ...')
        for data in raw_data:
            text.append(data['text'])
            spo_list.append(data['spo_list'])
           
        return (text, spo_list)
        
    def __compute_max_len(self, raw_data: list):

        max_seq_len = 0
        subject_max_len = 0
        for data in raw_data:
            max_seq_len = max(len(data['text']), max_seq_len)
            subject_max_len = max(max([len(spo['subject']) for spo in data['spo_list']]), subject_max_len)

        return max_seq_len, subject_max_len

    def get_inupts_outputs(self):
        return self.inputs_outputs

    def __len__(self):
        return self.len


class SpoDataset(Dataset):
    def __init__(self, inputs_outputs: list, max_seq_len: int):
        '''
        '''
        super(SpoDataset, self).__init__()
        text, spo_list = inputs_outputs

        self.text = text
        self.spo_list = spo_list
        self.max_seq_len = max_seq_len
        

        self.len = len(text)

    def __getitem__(self, index):
        
        text = self.text[index]
        spo_list = self.spo_list[index]

        max_len = self.max_seq_len

        s_start = [0.0  for _ in range(max_len)]
        s_end =  [0.0  for _ in range(max_len)]

        for spo in spo_list:
            s_start[spo['subject_start']] = 1.0
            s_end[spo['subject_end'] - 1] = 1.0
    
        return text, s_start, s_end, len(text)

    def __len__(self):
        return self.len

def collate_fn(data):
    '''
    '''
    
    lens = [item[3] for item in data]
    max_len = max(lens)

    text = [item[0] for item in data]

    s_start = [item[1][0: max_len] for item in data]
    s_end = [item[2][0: max_len] for item in data]

    as_tensor = torch.as_tensor

    ret = {
        'text': text,
        's_start': as_tensor(s_start),
        's_end': as_tensor(s_end),
    }

    return ret 

# 预测主实体的模型
class SubjectModel(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, rnn_hidden_size: int, forward_dim: int, device: str='cuda', dropout_prob: float=0.1):
        '''
        embedding_size： 词向量的大小
        num_predicate: 模型要预测的关系总数
        '''
        super(SubjectModel, self).__init__()
        
        self.embedding_processor = EmbeddingProcessor(embedding_dim=embedding_size, dropout_prob=dropout_prob)

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type='gru',
            hidden_size=rnn_hidden_size,
        )

        self.layernorm = nn.LayerNorm((embedding_size))

        self.attention = SelfAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        d = [1, 2, 3, 1, 2, 3]
        k = [5, 5, 5, 3, 3, 3]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )

        forward_dim_in = embedding_size * 1
        self.s_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 1),
        )

        self.s_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 1),
        )

    def forward(self, input_embedding: Tensor, mask: Tensor, position_embedding: Tensor):
        '''
        '''
        input_embedding = self.embedding_processor(input_embedding, position_embedding)

        share_feature = self.embedding_encoder(
            input_embedding=input_embedding,
            mask=mask,
        )

        share_feature = self.layernorm(share_feature + position_embedding)

        outs, _ = self.attention(
            inputs=share_feature,
            mask=mask,
        )
        outs = self.cnn(outs)

        s_start = self.s_start_fc(outs)
        s_end = self.s_end_fc(outs)
        
        return s_start, s_end

class Trainer(object):
    def __init__(self):
        '''
        sp_o模型训练
        '''
        super().__init__()
        self.train_data = TextData(TRAIN_FILE, ID_PREDICATE_FILE)
        self.dev_data = read_json(TEST_FILE)  # 评估数据直接用原始数据

        # 统一使用训练集的长度
        self.max_seq_len = self.train_data.max_seq_len
        
    def train(self, config : Config, device):

        # 训练数据加载器
        train_data_loader = DataLoader(
            dataset=SpoDataset(
                inputs_outputs=self.train_data.inputs_outputs,
                max_seq_len=self.max_seq_len,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,  # win10下，DataLoader使用lmdb进行多进程加载词向量会报错
            collate_fn=collate_fn, # 不自己写collate_fn处理numpy到tensor的转换会导致数据转换非常慢
            pin_memory=True,
        )

        # 报存模型的路径
        base_path = parent_path + '/model_file'
        pertrain_file = base_path + '/pertrain/merge_sgns_bigram_char300.npy'

        # torch_embedding
        embedding = None
        if config.from_pertrained == 'word2vec':
            config.embedding_size = 300
            embedding = TorchEmbedding(config.embedding_size, device,from_pertrain=pertrain_file).to(device)
        elif config.from_pertrained == 'albert':
            config.embedding_size = 768
            embedding = AlbertEmbedding(device=config.bert_device)
        elif config.from_pertrained == 'bert':
            config.embedding_size = 768
            embedding = Bert_embedding(device=config.bert_device)
        elif config.from_pertrained == 'none':
            embedding = TorchEmbedding(config.embedding_size, device).to(device)
        else:
            raise ValueError('value config.from_pertrained "{}" error.'.format(config.from_pertrained))
        
        if config.embedding_size % config.num_heads != 0:
            raise ValueError(('隐藏层的维度({})不是注意力头个数({})的整数倍'.format(config.embedding_size, config.num_heads)))

        position_embedding = PositionEmbedding(config.embedding_size).to(device)

        s_model = SubjectModel(
            embedding_size=config.embedding_size,
            num_heads=config.num_heads,
            rnn_hidden_size=config.rnn_hidden_size,
            forward_dim=config.forward_dim,
            device=device
        ).to(device)

        s_ema = EMA(model=s_model, decay=0.999)
      
        # sp_model.apply(init_weights)
        # s_model.apply(init_weights)

        # bce_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        bce_loss = FocalLossBCE(alpha=0.25, gamma=2.0, with_logits=True, device=device).to(device)


        # 网络参数
        params = []
        if config.from_pertrained not in ['bert', 'albert']:
            params.extend(get_models_parameters([embedding], weight_decay=0.0))
        params.extend(get_models_parameters(model_list=[s_model], weight_decay=0.0))
        

        # 优化器
        optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

        steps = int(np.round(self.train_data.len / config.batch_size))
        info = 'epoch: {}, steps: {}'.format(config.epoch, steps)
        print(info)

        best_f1 = 0.0
        best_epoch = 0
        patience = 10

        # warm_up_lambda是一个倍数算子
        warm_up_steps = int(steps * config.warm_up_epoch)
        warm_up_lambda = lambda step: (step + 1) / warm_up_steps
        warm_up_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_lambda)
        
        # 非等间隔余弦退火
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_T_max)


        # f1不上升的次数
        f1_not_up_count = 0
        f1 = 0.0
        loss_sum = 0.0
        loss_cpu = 0.0

        model_name = '{}_wv{}_{}{}'.format(config.from_pertrained, config.embedding_size, config.rnn_type, config.rnn_hidden_size)


        for epoch in range(config.epoch):
            s_model.train()
            log.info('epoch: {}, learning rate: {:.6f}, average loss: {:.6f}'.format(
                epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss_sum / steps
                )
            )
            loss_sum = 0.0
            with tqdm(total=steps) as pbar:
                for step, inputs_outputs in enumerate(train_data_loader):
                    pbar.update(1)
                    pbar.set_description('training epoch {}'.format(epoch))
                    pbar.set_postfix_str('loss: {:0.4f}'.format(loss_cpu))

                    text = inputs_outputs['text']
                   
                    s_start_true = inputs_outputs['s_start'].to(device)
                    s_end_true = inputs_outputs['s_end'].to(device)
                  
            
                    # 字符转向量
                    if config.from_pertrained in ['bert', 'albert']:
                        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
                    else:    
                        input_embedding, input_length = embedding(text)
                     
                    input_pos_embedding = position_embedding(input_length)
                    input_mask = create_mask_from_lengths(input_length)

                    s_start_pred, s_end_pred = s_model(
                        input_embedding=input_embedding,
                        mask=input_mask,
                        position_embedding=input_pos_embedding,
                    )
                
                    if config.from_pertrained in ['bert', 'albert']:
                        s_start_pred = s_start_pred[:, 1: - 1, :]
                        s_end_pred = s_end_pred[:, 1: - 1, :]
                        input_length -= 2
                        input_mask = create_mask_from_lengths(input_length)

                    s_start_pred = torch.squeeze(s_start_pred, dim=2)
                    s_end_pred = torch.squeeze(s_end_pred, dim=2)

                    loss_mask = torch.ge(input_mask, 1)

                    # 计算s模型的损失， mask掉的损失不做处理
                    s_start_loss =torch.masked_select(bce_loss(s_start_pred, s_start_true), loss_mask)
                    s_end_loss = torch.masked_select(bce_loss(s_end_pred, s_end_true), loss_mask)
                    s_start_loss = torch.mean(s_start_loss)
                    s_end_loss = torch.mean(s_end_loss)
                    
                    # 计算总的损失
                    loss = s_start_loss + s_end_loss 
                    loss_cpu = loss.cpu().detach().numpy()
                    loss_sum += loss_cpu

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters=s_model.parameters(), max_norm=10, norm_type=2.0)
                    optimizer.step()

                    # apply ema
                    s_ema.update_params()
            
                    if config.log_loss and (step % 100 == 0 or step == steps - 1) :
                        log.info('epoch: {}, step: {}, loss: {:.5f}.'.format(epoch, step, loss_cpu))

                    # warm up
                    if epoch < config.warm_up_epoch:
                        warm_up_schedule.step()
                    
                # end data loader
            # end process bar

            s_model.eval()
            s_ema.apply_shadow()

            print('{}, evaluate epoch: {} ...'.format(get_formated_time(), epoch))

            with torch.no_grad():
                f1, precision, recall = evaluate(
                    model=s_model,
                    embeddings=(embedding, position_embedding),
                    dev_data=self.dev_data,
                    config=config,
                )
                
            restore_best_state = False
            if f1 >= best_f1:
                best_f1 = f1
                best_epoch = epoch
                f1_not_up_count = 0
                s_ema.save_best_params()
                if config.from_pertrained not in ['bert', 'albert']:
                    torch.save(embedding.state_dict(), '{}/{}_s_embedding.22.pkl'.format(base_path, model_name))
                torch.save(s_model.state_dict(), '{}/{}_s_model.22.pkl'.format(base_path, model_name))
            else:
                f1_not_up_count += 1
                if f1_not_up_count >= patience:
                    info = 'f1 do not increase {} times, restore best state...'.format(patience)
                    print_and_log(info, log)
                    s_ema.restore_best_params()
                    restore_best_state = True
                    f1_not_up_count = 0
                    # patience += 1

            if not restore_best_state:
                s_ema.restore()
            
            info = 'epoch: {}, f1: {:.5f}, precision: {:.5f}, recall: {:.5f}, best_f1: {:.5f}, best_epoch: {}'.format(
                    epoch, f1, precision, recall, best_f1, best_epoch)
            print_and_log(info, log)
            
            # 调整学习率
            # Note that step should be called after validate()
            if epoch > config.warm_up_epoch:
                lr_scheduler.step()


def evaluate(model: SubjectModel, embeddings: tuple, dev_data: list, config: Config):
    '''
    评估
    '''
    # 最小的评估batch是64
    batch_size = config.batch_size if config.batch_size >= 64 else 64

    s_list_true = []
    s_list_pred = []
    batch_text = []
    n_dev_data = len(dev_data)

    for index, data in tqdm(enumerate(dev_data), total=n_dev_data):
        # 取出当前文本的真实spo
        current_s = []
        for spo in data['spo_list']:
            current_s.append(spo['subject'])
        s_list_true.append(current_s)

        batch_text.append(data['text'])

        if len(batch_text) == batch_size or index == n_dev_data - 1:
            batch_s = compute_batch_subject(
                model=model,
                embeddings=embeddings,
                text=batch_text,
                config=config,
            )
            s_list_pred.extend(batch_s)
            batch_text = []

    f1, precision, recall = f1_p_r_compute(s_list_pred, s_list_true, repair=False)

    return f1, precision, recall

def compute_batch_subject(model: SubjectModel, embeddings: tuple, text: list, config: Config):
    embedding, position_embedding = embeddings
    sigmoid = torch.sigmoid

    if config.from_pertrained in ['bert', 'albert']:
        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
    else:    
        input_embedding, input_length = embedding(text)
    input_pos_embedding = position_embedding(input_length)
    input_mask = create_mask_from_lengths(input_length)

    s_start_preds, s_end_preds = model(
        input_embedding=input_embedding,
        mask=input_mask,
        position_embedding=input_pos_embedding,
    )

    if config.from_pertrained in ['bert', 'albert']:
        s_start_preds = s_start_preds[:, 1: - 1, :]
        s_end_preds = s_end_preds[:, 1: - 1, :]

    s_start_preds = torch.squeeze(s_start_preds, dim=2)
    s_end_preds = torch.squeeze(s_end_preds, dim=2)

    s_start_preds = sigmoid(s_start_preds).cpu().detach().numpy()
    s_end_preds = sigmoid(s_end_preds).cpu().detach().numpy()
    
    batch_subject = []
    for text_, s_start_pred, s_end_pred in zip(text, s_start_preds, s_end_preds):
        # 可能有多个s
     
        s_start = s_start_pred[0: len(text_)]
        s_end = s_end_pred[0: len(text_)]
        current_text_s = []
        for i, s_s in enumerate(s_start):
            if s_s >= 0.45:
                for j, s_e in enumerate(s_end[i: ]):
                    if s_e >= 0.4:
                        subject_ = text_[i: i + j + 1]
                        current_text_s.append(subject_)
                        break
        batch_subject.append(current_text_s)
    
    return batch_subject

def load_model_and_test(config: Config, device):
    base_path = parent_path + '/model_file'
    dev_data = read_json(TEST_FILE) 

    embedding = TorchEmbedding(config.embedding_size, device).to(device)
    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    s_model = SubjectModel(
        embedding_size=config.embedding_size,
        num_heads=config.num_heads,
        forward_dim=config.forward_dim,
        rnn_hidden_size=config.rnn_hidden_size,
        device=device
    ).to(device)

    model_name = '{}_wv{}_{}{}'.format(config.from_pertrained, config.embedding_size, config.rnn_type, config.rnn_hidden_size)
    embedding.load_state_dict(torch.load('{}/{}_s_embedding.22.pkl'.format(base_path, model_name), map_location='cuda:0'))
    s_model.load_state_dict(torch.load('{}/{}_s_model.22.pkl'.format(base_path, model_name), map_location='cuda:0'))
   
    s_model.eval()
   

    with torch.no_grad():
        f1, precision, recall = evaluate(
                model=s_model, 
                embeddings=(embedding, position_embedding),
                dev_data=dev_data,
                config=config,
                )

    print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))