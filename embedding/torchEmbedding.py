from os.path import dirname, abspath
import torch 
import torch.nn as nn
import ujson
import codecs
import numpy as np
from torch import Tensor
from typing import Union

parent_path = abspath(dirname(dirname(__file__))) + '/data/'
CHAR2ID_FILE = parent_path + 'char2id.json'

# bert vocab
TORCH_BERT_DIR = parent_path + '/model_file/bert'
BERT_VOCAB = TORCH_BERT_DIR + '/vocab.txt'

# 位置编码
class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size: int, max_seq_len: int = 300):
        '''
        位置编码层，该层没有要训练的参数
        max_sep_len: 句子的最大长度，或者对齐（padding）的长度
        embedding_size： 一个词对应的词向量大小
        '''
        super(PositionEmbedding, self).__init__()

        # 根据论文公式构造PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, (2.0 * i) / embedding_size) for i in range(embedding_size)] \
                for pos in range(max_seq_len)
        ])

        # 奇数列用cos处理，偶数列用sin处理
        position_encoding[:, 1:: 2] = np.cos( position_encoding[:, 1:: 2] )
        position_encoding[:, 0:: 2] = np.sin( position_encoding[:, 0:: 2] )
        
        # 填充字符'[PAD]'position_encoding[0]的位置
        pad_encoding = torch.zeros((1, embedding_size), dtype=torch.float32)
        position_encoding = torch.cat( (pad_encoding, torch.Tensor(position_encoding)) )

        # num_embeddings= max_sep_len + 1, +1是因为添加了一个填充字符的编码
        # embedding_dim= embedding_size
        self.position_embedding = nn.Embedding(max_seq_len + 1, embedding_size)
        self.position_embedding.weight = nn.Parameter(position_encoding, requires_grad=False)
        self.max_seq_len = max_seq_len

    def forward(self, sequence_len, is_mask: bool = False, max_len: int=None):
        '''
        is_mask = False:
            sequence_len: [batch_size,], int or long, 一个batch中每个句子的真实长度，不包括填充的长度
            如：[1,5,7]
        
        设置is_mask = True表示传入的sequence_len是mask
        is_maks = True:
            sequence_len: [batch_size, padded_len or max_seq_len]
            如：[[1, 1, 0],[1, 0, 0]]
        '''
        mask = None
        if is_mask:
            mask = sequence_len
            sequence_len = torch.sum(sequence_len, dim=1, keepdim=True, dtype=torch.long)
        
        if max_len is None:
            max_len = torch.max(sequence_len)

        # range从1开始是因为embedding[0]放置的是pad的位置向量
        sequence_len_cpu = sequence_len.cpu().detach().numpy()
        input_id = [list(range(1, seq_len + 1)) + [0] * (max_len - seq_len) for seq_len in sequence_len_cpu]
        input_id = torch.LongTensor(input_id).to(sequence_len.device)
        
        outputs = self.position_embedding(input_id)

        if is_mask:
            mask = torch.unsqueeze(mask, dim=2)
            outputs = torch.mul(outputs, mask)
        
        return outputs

class TorchEmbedding(nn.Module):
    def __init__(self, embedding_size: int, device: str='cuda', char2id_file: str=CHAR2ID_FILE, from_pertrain: str=None):
        '''
            embedding_size: 词向量大小
            char2id_file: char2id的json文件
        '''
        super(TorchEmbedding, self).__init__()
        
        self.device = device

        # [pad]: 0, [unk]: 1,
        with codecs.open(char2id_file, 'r', encoding='utf-8') as f:
            self.char2id = ujson.load(f)

        if from_pertrain is not None:
            embedding_init = np.load(from_pertrain).astype(np.float32)
            embedding_init =  torch.FloatTensor(embedding_init)
            self.embedding = nn.Embedding.from_pretrained(embedding_init, freeze=False, padding_idx=0).to(device)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=len(self.char2id),
                embedding_dim=embedding_size,
                padding_idx=0,  # 填充的id是0
            ).to(device)

    def tokenize(self, text: list, max_len: int=None):
        '''
        args:
            text: 一个n个文本的list
            max_len: 文本的最大对齐长度
        return :
            text_id: 文本中每个字符对应的id
            mask: 长度和max_len一直，有字符的位置为1.0，填充的字符为0.0
        '''
        # [pad]: 0, [unk]: 1,

        length = np.array([len(text) for text in text], dtype=np.int64)
        batch_size = len(text)

        if max_len is None:
            max_len = max(length)

        text_id = np.zeros((batch_size, max_len), dtype=np.int64)
        get = self.char2id.get

        for i in range(batch_size):
            for j, char in enumerate(text[i]):
                text_id[i][j] = get(char, 1) # 没有找到字符返回1，1是unk

        text_id = torch.from_numpy(text_id).to(self.device)
        length = torch.from_numpy(length).to(self.device)

        return text_id, length

    def text_to_embedding(self, text: Union[list, str], max_len: int=None):
        '''
        多个文本转向量
        return:
            text_embedding: [batch_size, max_len, embedding_size]
            length: [batch_size, 1]
        '''
        return self.forward(text, max_len)
    
    def forward(self, text: Union[list, str], max_len: int=None, requires_grad: bool=True):
        if isinstance(text, str):
            text = [text]
        
        input_id, text_length = self.tokenize(text, max_len=max_len)
        text_embedding = self.embedding(input_id)

        if not requires_grad:
            text_embedding = text_embedding.detach()

        return text_embedding, text_length

class ClassEmbedding(nn.Module):
    def __init__(self, num_class: int, embedding_size: int):
        super(ClassEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=embedding_size)

    def forward(self, class_id: Tensor):
        return self.embedding(class_id)

class EmbeddingProcessor(nn.Module):
    def __init__(self, embedding_dim: int, dropout_prob: float=0.25):
        super(EmbeddingProcessor, self).__init__()
        
        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.embedding_layer_norm = nn.LayerNorm((embedding_dim))

    def forward(self, input_embedding: Tensor, position_embedding: Tensor=None):

        if position_embedding is not None:
            input_embedding = input_embedding + position_embedding
        
        input_embedding = self.embedding_dropout(input_embedding)
        outs = self.embedding_layer_norm(input_embedding)

        return outs

if __name__ == "__main__":

    embedding = ClassEmbedding(3, 4)
    c_id = torch.LongTensor([1,0,1,2])
    emb = embedding(c_id)
    print(emb.shape)
    print(emb)

    exit()
    # mask = torch.LongTensor([
    #     [1, 1, 1, 1, 0, 0],
    #     [1, 1, 1, 1, 1, 0]
    # ])
    # position_embedding = PositionEmbedding(embedding_size=4)
    # pos_emb = position_embedding(mask, is_mask=True, max_len=6)
    # print(pos_emb.shape)
    # print(pos_emb.dtype)


    embedding = TorchEmbedding(embedding_size=4, device='cpu')
    proc = EmbeddingProcessor(embedding_dim=4)

    text = ['你好吗？','你 好']
    
    emb, length = embedding(text, requires_grad=False)
    print(length)
    print(emb.shape)
    print(emb)
    print(proc(emb))
    