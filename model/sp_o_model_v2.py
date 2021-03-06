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
from torch.nn.modules.activation import GELU
from torch.nn.modules.linear import Linear
from tqdm import tqdm

sys.path.append('..')
sys.path.append('.')
from utils.function import *
from utils.logger import Logger
from embedding.torchEmbedding import TorchEmbedding, PositionEmbedding, EmbeddingProcessor
from embedding.AlbertEmbedding import AlbertEmbedding
from model.attention import *
from model.Decoder import DGCNNDecoder, LinearDecoder
from model.Encoder import RNNEncoder
from model.model_utils import EMA
from model.loss_function import FocalLossBCE
from config import Config

log = Logger('sp_o_model_v2', std_out=False, save2file=True).get_logger()

parent_path = abspath(dirname(dirname(__file__)))
TRAIN_FILE = parent_path + '/data/my_train_data.json'
DEV_FILE = parent_path + '/data/my_dev_data.json'
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
        
        self.predicate_info = read_json(PREDICATE_INFO_FILE)
        self.id2predicate, self.predicate2id = read_json(id_predicate_file)
        with codecs.open(char2id_file, 'r', encoding='utf-8') as f:
            self.char2id = ujson.load(f)

        # 关系总数
        # 0是unk类
        self.num_predicate = len(self.id2predicate)
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

            # 随机采样一个sp作为o模型的输入
            choice_index.append(randint(low=0, high=len(data['spo_list'])))
           
        return (text, spo_list, choice_index)
        
    def __compute_max_len(self, raw_data: list):

        max_seq_len = 0
        subject_max_len = 0
        for data in raw_data:
            max_seq_len = max(len(data['text']), max_seq_len)
            subject_max_len = max(max([ len(spo['subject']) for spo in data['spo_list']]), subject_max_len)

        return max_seq_len, subject_max_len

    def get_inupts_outputs(self):
        return self.inputs_outputs

    def __len__(self):
        return self.len


class SpoDataset(Dataset):
    def __init__(self, inputs_outputs: list, predicate_info: dict, predicate2id: dict, max_seq_len: int):
        '''
        '''
        super(SpoDataset, self).__init__()
        text, spo_list, choice_index = inputs_outputs

        self.text = text
        self.spo_list = spo_list
        self.choice_index = choice_index

        self.predicate_info = predicate_info
        self.max_seq_len = max_seq_len
        self.num_predicate = len(predicate2id)
        self.predicate2id = predicate2id

        self.len = len(text)

    def __getitem__(self, index):
        
        text = self.text[index]
        spo_list = self.spo_list[index]
        choice_index = self.choice_index[index]

        max_len = self.max_seq_len
        predicate2id = self.predicate2id
        spo_choose = spo_list[choice_index]
        s_choose = spo_choose['subject']
        p_choose = spo_choose['predicate']
        p_id = predicate2id[p_choose]

        sp_start = [[0.0] * self.num_predicate for _ in range(max_len)]
        sp_end = [[0.0] * self.num_predicate for _ in range(max_len)]
        o_start = [0] * max_len
        o_end = [0] * max_len

        s_choose_start_end = [spo_choose['subject_start'], spo_choose['subject_end'] - 1]

        for spo in spo_list:
            predicate_id = predicate2id[spo['predicate']]
            sp_start[spo['subject_start']][predicate_id] = 1.0
            sp_end[spo['subject_end'] - 1][predicate_id] = 1.0

            if spo['subject'] == s_choose and spo['predicate'] == p_choose:
                o_start[spo['object_start']] = 1
                o_end[spo['object_end'] - 1] = 1

       
        return text, sp_start, sp_end, o_start, o_end, s_choose_start_end, p_id, len(text)

    def __len__(self):
        return self.len

def collate_fn(data):
    '''
    '''
    array = np.array
    
    lens = [item[7] for item in data]
    max_len = max(lens)

    text = [item[0] for item in data]

    sp_start = [item[1][0: max_len] for item in data]
    sp_end = [item[2][0: max_len] for item in data]

    o_start = [item[3][0: max_len] for item in data]
    o_end = [item[4][0: max_len] for item in data]

    s_start_end = array([item[5] for item in data])
    p_id = [item[6] for item in data]

    as_tensor = torch.as_tensor

    ret = {
        'text': text,
        'sp_start': as_tensor(sp_start),
        'sp_end': as_tensor(sp_end),
        'o_start': as_tensor(o_start).long(),
        'o_end': as_tensor(o_end).long(),
        's_start_end': s_start_end,
        'p_id': as_tensor(p_id).long()
    }

    return ret 

# 预测主实体的模型
class SubjectPredicateModel(nn.Module):
    def __init__(self, embedding_size: int, num_predicate: int, num_heads: int, rnn_hidden_size: int, forward_dim: int=128, device: str='cuda', dropout_prob: float=0.1):
        '''
        embedding_size： 词向量的大小
        num_predicate: 模型要预测的关系总数
        '''
        super(SubjectPredicateModel, self).__init__()
        
        self.embedding_processor = EmbeddingProcessor(embedding_dim=embedding_size, dropout_prob=dropout_prob)

        self.embedding_encoder = RNNEncoder(
            embedding_dim=embedding_size,
            num_layers=2,
            rnn_type='gru',
            hidden_size=rnn_hidden_size,
        )

        self.self_attention = SelfAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        d = [1, 2, 3, 4]
        k = [5, 3, 3, 3]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )

        forward_dim_in = embedding_size

        self.weight_fc = nn.Sequential(
            nn.Linear(forward_dim_in, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 1),
            nn.Sigmoid()
        )

        self.start_weight_fc = nn.Sequential(
            nn.Linear(forward_dim_in, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, num_predicate),
            nn.Sigmoid()
        )

        self.end_weight_fc = nn.Sequential(
            nn.Linear(forward_dim_in, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, num_predicate),
            nn.Sigmoid()
        )

        self.sp_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, num_predicate)
        )
        self.sp_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, num_predicate)
        )


    def forward(self, input_embedding: Tensor, mask: Tensor, position_embedding: Tensor):
        '''
        '''
        input_embedding = self.embedding_processor(input_embedding, position_embedding)

        share_feature = self.embedding_encoder(
            input_embedding=input_embedding,
            mask=mask,
        )

        #===========================================================================================#
        cnn_outs = self.cnn(share_feature, mask)

        outs, _ = self.self_attention(
            inputs=cnn_outs,
            mask=mask,
        )
        
        weight = self.weight_fc(outs)

        start_weight = self.start_weight_fc(outs)
        end_weight = self.end_weight_fc(outs)
        
        sp_start = self.sp_start_fc(outs)
        sp_end = self.sp_end_fc(outs)

        sp_start = sp_start * start_weight * weight
        sp_end = sp_end * end_weight * weight
        
        return share_feature, sp_start, sp_end, weight

class ObjectModel(nn.Module):
    def __init__(self, embedding_size: int, num_predicate: int, num_heads: int, forward_dim: int=128, device: str='cuda'):
        '''
        '''
        super(ObjectModel, self).__init__()

        self.predicate_embedding = nn.Embedding(num_embeddings=num_predicate, embedding_dim=embedding_size)

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            device=device,
        )

        d = [1, 2, 3, 4]
        k = [5, 3, 3, 3]
        self.cnn = DGCNNDecoder(
            dilations=d,
            kernel_sizes=k,
            in_channels=embedding_size,
        )


        forward_dim_in = embedding_size

        self.start_weight_fc = nn.Sequential(
            nn.Linear(forward_dim_in, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 2),
            nn.Sigmoid()
        )

        self.end_weight_fc = nn.Sequential(
            nn.Linear(forward_dim_in, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, 2),
            nn.Sigmoid()
        )

        self.subject_start_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 2),
        )

        self.subject_end_fc = nn.Sequential(
            nn.Linear(in_features=forward_dim_in, out_features=forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, 2),
        )

    def forward(self, share_feature: Tensor, share_mask: Tensor, p_id: Tensor, s_start_end: np.array, sp_weight: Tensor):
        '''
        '''
        
        p_embedding = self.predicate_embedding(p_id)
        p_embedding = torch.unsqueeze(p_embedding, dim=1)

        inter_feature = tensor_gather(share_feature, s_start_end, max_n=16)
        inter_feature = inter_feature + p_embedding

        cnn_outs = self.cnn(share_feature, share_mask)

        outs, _ = self.multi_head_attention(
            query=cnn_outs,
            key=inter_feature,
            value=inter_feature,
            query_mask=share_mask,
        )

        start_weight = self.start_weight_fc(outs)
        end_weight = self.end_weight_fc(outs)

        subject_start = self.subject_start_fc(outs)
        subject_end = self.subject_end_fc(outs)

        subject_start = subject_start * start_weight * sp_weight
        subject_end = subject_end * end_weight * sp_weight

        return subject_start, subject_end

# ======================================================================= end model ======================================================================#

class Trainer(object):
    def __init__(self):
        '''
        sp_o模型训练
        '''
        super().__init__()
        self.train_data = TextData(TRAIN_FILE, ID_PREDICATE_FILE)
        self.dev_data = read_json(DEV_FILE)  # 评估数据直接用原始数据

        # 统一使用训练集的长度
        self.max_seq_len = self.train_data.max_seq_len
        self.num_predicate = self.train_data.num_predicate
    
        self.id2predicate = self.train_data.id2predicate
        self.predicate2id = self.train_data.predicate2id
        self.predicate_info = self.train_data.predicate_info

    def train(self, config: Config, device):

        # 训练数据加载器
        train_data_loader = DataLoader(
            dataset=SpoDataset(
                inputs_outputs=self.train_data.inputs_outputs,
                predicate_info=self.predicate_info,
                predicate2id=self.predicate2id,
                max_seq_len=self.max_seq_len,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=1,  # win10下，DataLoader使用lmdb进行多进程加载词向量会报错
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

        sp_model = SubjectPredicateModel(
            embedding_size=config.embedding_size,
            num_predicate=self.num_predicate,
            num_heads=config.num_heads,
            forward_dim=config.forward_dim,
            rnn_hidden_size=config.rnn_hidden_size,
            device=device
        ).to(device)

        o_model = ObjectModel(
            embedding_size=config.embedding_size,
            num_predicate=self.num_predicate,
            num_heads=config.num_heads,
            forward_dim=config.forward_dim,
            device=device,
        ).to(device)

        sp_ema = EMA(model=sp_model, decay=0.9999)
        o_ema = EMA(model=o_model, decay=0.9999)

        # sp_model.apply(init_weights)
        # o_model.apply(init_weights)

        bce_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        ce_loss  = nn.CrossEntropyLoss(reduction='none').to(device)
        # bce_loss = FocalLossBCE(device=device, alpha=0.4, gamma=1.0, with_logits=True).to(device)
        # ce_loss = FocalLossCrossEntropy(num_class=self.num_predicate, device=device, alpha=0.9, gamma=2.0, reduce='none').to(device)

        # 网络参数
        params = []
        if config.from_pertrained not in ['bert', 'albert']:
            params.extend(get_models_parameters([embedding], weight_decay=0.0))
        params.extend(get_models_parameters(model_list=[sp_model, o_model], weight_decay=0.0))
        

        # 优化器
        optimizer = torch.optim.Adam(params=params, lr=config.learning_rate)

        steps = int(np.round(self.train_data.len / config.batch_size))
        info = 'epoch: {}, steps: {}'.format(config.epoch, steps)
        log.info('\n{}'.format('=' * 128))
        log.info(info)
        print(info)

        best_f1 = 0.0
        best_epoch = 0

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_T_max)

        # warm_up_lambda是一个倍数算子
        warm_up_steps = int(steps * config.warm_up_epoch)
        warm_up_lambda = lambda step: (step + 1) / warm_up_steps
        warm_up_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_lambda)
        # 非等间隔余弦退火
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        patience = 3
        f1_not_up_count = 0
        f1 = 0.0
        for epoch in range(config.epoch):
            sp_model.train()
            o_model.train()
            loss_sum = None
            loss_cpu = 0.0
            log.info('epoch: {}, lr: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            print('epoch: {}, start time: {}'.format(epoch, get_formated_time()))

            with tqdm(total=steps) as pbar:
                for step, inputs_outputs in enumerate(train_data_loader):
                    pbar.update(1)
                    pbar.set_description('training epoch {}'.format(epoch))
                    pbar.set_postfix_str('loss: {:0.4f}'.format(loss_cpu))

                    text = inputs_outputs['text']
                
                    sp_start_true = inputs_outputs['sp_start'].to(device)
                    sp_end_true = inputs_outputs['sp_end'].to(device)
                    o_start_true = inputs_outputs['o_start'].to(device)
                    o_end_true = inputs_outputs['o_end'].to(device)

                    p_id = inputs_outputs['p_id'].to(device)
                    s_start_end = inputs_outputs['s_start_end']
            
                    # 字符转向量
                    if config.from_pertrained in ['bert', 'albert']:
                        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
                    else:    
                        input_embedding, input_length = embedding(text)
                    input_pos_embedding = position_embedding(input_length)

                    input_mask = create_mask_from_lengths(input_length)

                    share_feature, sp_start_pred, sp_end_pred, weight = sp_model(
                        input_embedding=input_embedding,
                        mask=input_mask,
                        position_embedding=input_pos_embedding,
                    )

                    o_start_pred, o_end_pred = o_model(
                        share_feature=share_feature,
                        share_mask=input_mask,
                        p_id=p_id,
                        s_start_end=s_start_end,
                        sp_weight=weight,
                    )

                    # o_start_pred = torch.squeeze(o_start_pred, dim=2)
                    # o_end_pred = torch.squeeze(o_end_pred, dim=2)

                    o_start_pred = o_start_pred.permute(0, 2, 1)
                    o_end_pred = o_end_pred.permute(0, 2, 1)
                
                    if config.from_pertrained in ['bert', 'albert']:
                        sp_start_pred = sp_start_pred[:, 1: - 1, :]
                        sp_end_pred = sp_end_pred[:, 1: - 1, :]
                        o_start_pred = o_start_pred[:, 1: - 1]
                        o_end_pred = o_end_pred[:, 1: - 1]
                        input_length -= 2
                        input_mask = create_mask_from_lengths(input_length)

                    loss_mask = torch.ge(input_mask, 1)
                    sp_loss_mask = loss_mask.unsqueeze(dim=2)

                    # 计算sp模型的损失， mask掉的损失不做处理
                    sp_start_loss =torch.masked_select(bce_loss(sp_start_pred, sp_start_true), sp_loss_mask)
                    sp_end_loss = torch.masked_select(bce_loss(sp_end_pred, sp_end_true), sp_loss_mask)
                    sp_start_loss = torch.mean(sp_start_loss)
                    sp_end_loss = torch.mean(sp_end_loss)
                    
                    # 计算o模型损失
                    o_start_loss = torch.masked_select(ce_loss(o_start_pred, o_start_true), loss_mask)
                    o_end_loss = torch.masked_select(ce_loss(o_end_pred, o_end_true), loss_mask)
                    o_start_loss = torch.mean(o_start_loss)
                    o_end_loss = torch.mean(o_end_loss)
            
                    # 计算总的损失
                    loss_sum = 20.0 * (sp_start_loss + sp_end_loss) + 5.0 * (o_start_loss + o_end_loss)
                    loss_cpu = loss_sum.cpu().detach().numpy()

                    optimizer.zero_grad()
                    loss_sum.backward()
                    nn.utils.clip_grad_norm_(parameters=sp_model.parameters(), max_norm=10, norm_type=2.0)
                    nn.utils.clip_grad_norm_(parameters=o_model.parameters(), max_norm=10, norm_type=2.0)
                    optimizer.step()

                    # apply ema 
                    sp_ema.update_params()
                    o_ema.update_params()

                    if step % 100 == 0 or step == steps - 1:
                        log.info('epoch: {}, step: {}, loss: {:.5f} .'.format(epoch, step, loss_cpu))

                    # warm up
                    if epoch < config.warm_up_epoch:
                        warm_up_schedule.step()
                    
                # end data loader
            # end process bar

            sp_model.eval()
            o_model.eval()

            sp_ema.apply_shadow()
            o_ema.apply_shadow()

            print('{}, evaluate epoch: {} ...'.format(get_formated_time(), epoch))
            log.info('evaluate epoch: {} ...'.format(epoch))
            with torch.no_grad():
                f1, precision, recall, _ = evaluate(
                    models=(sp_model, o_model),
                    embeddings=(embedding, position_embedding),
                    dev_data=self.dev_data,
                    config=config,
                    id2predicate = self.id2predicate,
                )
            
            restore_best_state = False
            if f1 >= best_f1:
                best_f1 = f1
                best_epoch = epoch
                sp_ema.save_best_params()
                o_ema.save_best_params()
                if config.from_pertrained not in ['bert', 'albert']:
                    torch.save(embedding.state_dict(), '{}/{}v2_sp_o_embedding.pkl'.format(base_path, config.from_pertrained))
                torch.save(sp_model.state_dict(), '{}/{}v2_sp_model.pkl'.format(base_path, config.from_pertrained))
                torch.save(o_model.state_dict(), '{}/{}v2_o_model.pkl'.format(base_path, config.from_pertrained))
            else:
                f1_not_up_count += 1
                if f1_not_up_count >= patience:
                    info = 'f1 do not increase {} times, restore best state...'.format(patience)
                    print_and_log(info, log)
                    sp_ema.restore_best_params()
                    o_ema.restore_best_params()
                    restore_best_state = True
                    f1_not_up_count = 0
                    
                    # 恢复权重后要修改学习率
                    adjust_learning_rate(0.0001, optimizer)
                if f1 < 0.65:
                    adjust_learning_rate(0.001, optimizer)

            if not restore_best_state:
                sp_ema.restore()
                o_ema.restore()
            
            info = 'epoch: {}, f1: {:.5f}, precision: {:.5f}, recall: {:.5f}, best_f1: {:.5f}, best_epoch: {}'.format(
                    epoch, f1, precision, recall, best_f1, best_epoch)
            print_and_log('=' * 64, log)
            print_and_log(info, log)
            print_and_log('=' * 64, log)
            
            # 调整学习率
            # Note that step should be called after validate()
            lr_scheduler.step()


def evaluate(models: tuple, embeddings: tuple, dev_data: list, config: Config, id2predicate: dict, show_details: bool=False):
    '''
    评估
    '''
    # 最小的评估batch是64
    batch_size = config.batch_size if config.batch_size >= 64 else 64

    spo_list_true = []
    spo_list_pred = []
    batch_text = []
    n_dev_data = len(dev_data)

    for index, data in tqdm(enumerate(dev_data), total=n_dev_data):
        # 取出当前文本的真实spo
        current_spo = []
        for spo in data['spo_list']:
            current_spo.append( (spo['subject'], spo['predicate'], spo['object']) )
        spo_list_true.append(current_spo)

        batch_text.append(data['text'])

        if len(batch_text) == batch_size or index == n_dev_data - 1:
            batch_spo = compute_batch_spo(
                models=models,
                embeddings=embeddings,
                text=batch_text,
                id2predicate=id2predicate,
                config=config,
            )
            spo_list_pred.extend(batch_spo)
            batch_text = []

    if show_details: 
        def get_spo(spo_list_all: list):
            s_list, sp_list, p_list, o_list = [], [], [], []
            for spo_list in spo_list_all:
                current_s, current_sp, current_p, current_o = [], [], [], []
                for spo in spo_list:
                    current_s.append((spo[0],))
                    current_p.append((spo[1],))
                    current_o.append((spo[2], ))
                    current_sp.append((spo[0], spo[1]))

                s_list.append(current_s)
                p_list.append(current_p)
                o_list.append(current_o)
                sp_list.append(current_sp)
            return s_list, p_list, o_list, sp_list

        def show_f1_p_r(p: list, t: list, name: str):
            f1, precision, recall = f1_p_r_compute(p, t)
            info = '{}, f1: {:.5f}; precision: {:.5f}; recall: {:.5f} '.format(name ,f1, precision, recall)
            print_and_log(info, log)

        s_pred, p_pred, o_pred, sp_pred = get_spo(spo_list_pred)
        s_true, p_true, o_true, sp_true = get_spo(spo_list_true)
        show_f1_p_r(s_pred, s_true,'subject')
        show_f1_p_r(p_pred, p_true,'preidcate')
        show_f1_p_r(o_pred, o_true,'object')
        show_f1_p_r(sp_pred, sp_true,'subject and predicate')

    f1, precision, recall = f1_p_r_compute(spo_list_pred, spo_list_true)

    return f1, precision, recall, (spo_list_pred, spo_list_true)


def compute_batch_spo(models: tuple, embeddings: tuple, text: list, id2predicate: dict, config: Config):
    sp_model, o_model = models

    share_features, share_masks, sp_start_preds, sp_end_preds, weights = compute_batch_sp(
        sp_model=sp_model,
        embeddings=embeddings,
        text=text,
        config=config,
     )

    device =share_features.device
    
    batch_size = share_features.shape[0]
    batch_sp = []
    batch_share_feature = []
    batch_mask = []
    batch_p_id = []
    batch_s_start_end = []
    batch_weight = []
    batch_ids = []


    # 抽取一个批次的po
    for bs_id, (share_feature, share_mask, sp_start_pred, sp_end_pred, weight) in enumerate(zip(share_features, share_masks, sp_start_preds, sp_end_preds, weights)):
        text_ = text[bs_id]
        sp_start = np.where(sp_start_pred[0: len(text_)] >= 0.4)
        sp_end = np.where(sp_end_pred[0: len(text_)] >= 0.4)
        for s_start, s_p_id in zip(*sp_start):
            for s_end, e_p_id in zip(*sp_end):
                if s_start <= s_end and s_p_id == e_p_id:
                    # 发现一个sp
                    subject = text_[s_start: s_end + 1]
                    predicate = id2predicate[str(s_p_id)]


                    batch_ids.append(bs_id)     # 记录po是那一条text的

                    batch_sp.append((subject, predicate))
                    batch_share_feature.append(share_feature)
                    batch_mask.append(share_mask)
                    batch_weight.append(weight)
                    batch_p_id.append(s_p_id)
                    batch_s_start_end.append([s_start, s_end - 1])
                    
                    # 就近匹配，不再往下找了
                    break

    last_start = 0
    n = int(np.ceil(len(batch_mask) / batch_size))
    batch_spo_pred = [[] for _ in range(batch_size)]

    max_seq_len = share_features.shape[1]
    embedding_dim = share_features.shape[2]

    # 对一个batch中抽取到的所有po抽取s
    for _ in range(n):
        end = last_start + batch_size

        bs_share_feature = torch.cat(batch_share_feature[last_start: end], dim=0)
        # 直接cat结果不对，要重新reshape
        bs_share_feature = bs_share_feature.reshape(-1, max_seq_len, embedding_dim)

        bs_input_mask = batch_mask[last_start: end]
        bs_input_mask = torch.cat(bs_input_mask, dim=0)
        bs_input_mask = bs_input_mask.reshape(-1, max_seq_len)

        bs_weight = batch_weight[last_start: end]
        bs_weight = torch.cat(bs_weight, dim=0)

        bs_weight = bs_weight.reshape(-1, max_seq_len, 1)
 
        p_id = batch_p_id[last_start: end]
        p_id = torch.as_tensor(p_id).long().to(device)
        s_start_end = np.array(batch_s_start_end[last_start: end])

        o_start_preds, o_end_preds = compute_o(
            o_model=o_model,
            share_feature=bs_share_feature,
            share_mask=bs_input_mask,
            p_id=p_id,
            s_start_end=s_start_end,
            batch_weight=bs_weight,
            config=config,
        )

        for bs_id, sp, o_start_pred, o_end_pred in zip(batch_ids[last_start: end], batch_sp[last_start: end], o_start_preds, o_end_preds):
            # 可能有多个s
            text_ = text[bs_id]
            o_start = o_start_pred[0: len(text_)]
            o_end = o_end_pred[0: len(text_)]

            for i, o_s in enumerate(o_start):
                if o_s == 1:
                    for j, o_e in enumerate(o_end[i: ]):
                        if o_e >= 1:
                            object_ = text_[i: i + j + 1]
                            batch_spo_pred[bs_id].append(sp + (object_, ))
                            break
        # 更新last start
        last_start += batch_size
    
    return batch_spo_pred

def compute_batch_sp(sp_model: SubjectPredicateModel, embeddings: tuple, text: list, config: Config):
    '''
    '''
    embedding, position_embedding = embeddings
    sigmoid = torch.sigmoid

    if config.from_pertrained in ['bert', 'albert']:
        input_embedding, cls_embedding, input_length, attention_mask = embedding(text)
    else:    
        input_embedding, input_length = embedding(text)
    input_pos_embedding = position_embedding(input_length)
    input_mask = create_mask_from_lengths(input_length)

    share_feature, sp_start_pred, sp_end_pred, weight = sp_model(
        input_embedding=input_embedding,
        mask=input_mask,
        position_embedding=input_pos_embedding,
    )

    if config.from_pertrained in ['bert', 'albert']:
        sp_start_pred = sp_start_pred[:, 1: - 1, :]
        sp_end_pred = sp_end_pred[:, 1: - 1, :]

    sp_start_pred = sigmoid(sp_start_pred).cpu().detach().numpy()
    sp_end_pred = sigmoid(sp_end_pred).cpu().detach().numpy()
    
    return share_feature, input_mask, sp_start_pred, sp_end_pred, weight


def compute_o(o_model: ObjectModel, share_feature: Tensor, share_mask: Tensor, p_id: Tensor, s_start_end: np.array, batch_weight: Tensor, config: Config):
    '''
    '''
    argmax = torch.argmax

    o_start_pred, o_end_pred = o_model(
            share_feature=share_feature,
            share_mask=share_mask,
            p_id=p_id,
            s_start_end=s_start_end,
            sp_weight=batch_weight,
        )
    # o_start_pred = torch.squeeze(o_start_pred, dim=2)
    # o_end_pred = torch.squeeze(o_end_pred, dim=2)

    if config.from_pertrained in ['bert', 'albert']:
        o_start_pred = o_start_pred[:, 1: - 1]
        o_end_pred = o_end_pred[:, 1: - 1]

    o_start_pred = argmax(o_start_pred, dim=2).cpu().detach().numpy()
    o_end_pred = argmax(o_end_pred, dim=2).cpu().detach().numpy()
    return o_start_pred, o_end_pred

def load_model_and_evalute(config: Config, device):
    base_path = parent_path + '/model_file'
    dev_data = read_json(DEV_FILE) 

    embedding = TorchEmbedding(config.embedding_size, device).to(device)
    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    id2predicate, predicate2id = read_json(ID_PREDICATE_FILE)
    num_predicate = len(id2predicate)

    position_embedding = PositionEmbedding(config.embedding_size).to(device)

    sp_model = SubjectPredicateModel(
        embedding_size=config.embedding_size,
        num_predicate=num_predicate,
        num_heads=config.num_heads,
        forward_dim=config.forward_dim,
        rnn_hidden_size=config.rnn_hidden_size,
        device=device
    ).to(device)

    o_model = ObjectModel(
        embedding_size=config.embedding_size,
        num_predicate=num_predicate,
        num_heads=config.num_heads,
        forward_dim=config.forward_dim,
        device=device,
    ).to(device)

    embedding.load_state_dict(torch.load('{}/{}v2_sp_o_embedding.pkl'.format(base_path, config.from_pertrained), map_location='cuda:0'))
    sp_model.load_state_dict(torch.load('{}/{}v2_sp_model.pkl'.format(base_path, config.from_pertrained), map_location='cuda:0'))
    o_model.load_state_dict(torch.load('{}/{}v2_o_model.pkl'.format(base_path, config.from_pertrained), map_location='cuda:0'))

    sp_model.eval()
    o_model.eval()

    with torch.no_grad():
        f1, precision, recall, (spo_list_pred, spo_list_true) = evaluate(
                models=(sp_model, o_model),
                embeddings=(embedding, position_embedding),
                dev_data=dev_data,
                config=config,
                id2predicate=id2predicate,
                show_details=True,
                )
    save_spo_list(dev_data, spo_list_pred, spo_list_true, parent_path + '/data/spo_list_pred.json')

    print('f1: {:.5f}; precision: {:.5f}; recall: {:.5f}'.format(f1, precision, recall))