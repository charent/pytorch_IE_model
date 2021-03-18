import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import sys
sys.path.append('.')
sys.path.append('..')

from model.attention import SelfAttention, MultiHeadAttention

class SelfAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int, dropout_prob: float=0.0, device: str='cuda'):
        super(SelfAttentionEncoder, self).__init__()

        self.attention_list = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_list.append(
                SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    device=device,
                )
            )

    def forward(self, inputs: Tensor, mask: Tensor=None):
        
        for attention in self.attention_list:
            inputs, _ = attention(inputs, mask)

        return inputs

class MultiheadAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int, dropout_prob: float=0.0, device: str='cuda'):
        super(MultiheadAttentionEncoder, self).__init__()

        self.attention_list = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_list.append(
                MultiHeadAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                    device=device,
                )
            )

    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_mask: Tensor=None, key_mask:Tensor=None):
        
        for attention in self.attention_list:
            query, _ = attention(query=query, key=key, value=value, query_mask=query_mask, key_mask=key_mask)

        return query

class RNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int, hidden_size: int, rnn_type: str='gru', dropout_prob: float=0.1):
        '''
        rnn_type: 'lstm', 'rnn'
        '''
        super(RNNEncoder, self).__init__()

        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.embedding_layer_norm = nn.LayerNorm((embedding_dim))
        
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True,
                bidirectional=True
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                batch_first=True, 
                bidirectional=True
            )
        else:
            raise ValueError('value rnn_type({}) not in [\'lstm\', \'gru\']'.format(rnn_type))

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.ReLU(),
        )

        self.out_layer_norm = nn.LayerNorm((embedding_dim))

    def forward(self, input_embedding: Tensor, position_embedding: Tensor, mask: Tensor):
        '''
        '''
        self.rnn.flatten_parameters()
        
        max_seq_len = input_embedding.shape[1]

        if position_embedding is not None:
            input_embedding = input_embedding + position_embedding
        
        input_embedding = self.embedding_dropout(input_embedding)
        input_embedding = self.embedding_layer_norm(input_embedding)

        # 长度应该是cpu上的向量
        input_lengths_cpu = torch.sum(mask, dim=1).cpu()

        inputs = pack_padded_sequence(input=input_embedding, lengths=input_lengths_cpu, batch_first=True, enforce_sorted=False)
        inputs, h_x = self.rnn(inputs)
        out, _ = pad_packed_sequence(inputs, batch_first=True, total_length=max_seq_len)

        out = self.out_fc(out)

        out = self.out_layer_norm(out + input_embedding + position_embedding)

        return out

if __name__ == '__main__':

    device = 'cpu'
    test_encoder = True

    if test_encoder:
        x = torch.randn((2, 3, 4))
        pos = torch.randn((2, 3, 4))
        mask = torch.FloatTensor([[1,1,0],[1,1,1]])

        model = RNNEncoder(embedding_dim=4, num_layers=2, hidden_size=4, rnn_type='lstm')
        
        out = model(x, mask)
        print(out.shape)
        print(out)

    exit()
    x = torch.randn((2,3,4))
    mask = torch.LongTensor([[1,1,0],[1,1,1]])
    encoder = MultiheadAttentionEncoder(4, 2, 2, device='cpu')
    out = encoder(x,x,x, mask)
    print(out.shape)
    print(out)
    