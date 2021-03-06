import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

#  自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_prob: float=0.0, device: str='cuda'):
        '''
        自注意力机制层
        embedding_dim：词向量维度
        num_heads：注意力头的个数
        dropout_prob: dropout的概率
        '''
        super(SelfAttention, self).__init__()
       
        if embedding_dim % num_heads != 0:
            raise ValueError('隐藏层的维度({})不是注意力头个数({})的整数倍'.format(embedding_dim, num_heads))

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # 每个注意力头的维度
        self.head_dim = int(embedding_dim / num_heads)

        self.query = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.dropout = nn.Dropout(dropout_prob)

        self.out_fc = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
        )
        
        self.layer_norm = nn.LayerNorm((embedding_dim))
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]).to(device))

    def transpose_tensor(self, x: Tensor, batch_size):
        '''
        对x做变换
        '''
        # [batch_size, sequence_length, num_heads, head_dim]
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
  
        # [batch_size, num_heads, sequence_length, head_dim]
        x = x.permute(0, 2, 1, 3)

        return x

    def forward(self, inputs: Tensor, mask: Tensor=None, need_weights: bool=False):
        '''
        inputs.shape = [batch_size, sequence_length, embedding_dim], float
        mask.shape = [batch_size, sequence_length], 如[[1, 1, 0],[1, 0, 0]], flaot

        输出：
        outputs: [batch_size, seq_len, embedding_dim]
        '''
        batch_size = inputs.shape[0]

        # 残差
        residual = inputs

        # 对输入做线性变换
        Q = self.query(inputs) # [batch_size, seq_len, embedding_dim]
        K = self.key(inputs) # [batch_size, seq_len, embedding_dim]
        V = self.value(inputs) # [batch_size, seq_len, embedding_dim]

        # 把线下变换后得到的layer中embedding_dim维度拆分，放到第1，第3维度
        # = [batch_size, num_heads, sequence_length, head_dim]
        Q = self.transpose_tensor(Q, batch_size)
        K = self.transpose_tensor(K, batch_size)
        V = self.transpose_tensor(V, batch_size)

        # 通过点乘获取query和key之间的权重分数
        # 要除以scale，否则分数过大，会出现softmax结果非0既1
        # [batch_size, num_heads, sequence_length, head_dim] * [batch_size, num_heads, head_dim, sequence_length]
        # [batch_size, num_heads, seq_len, seq_len]
        energy  = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is None:
            #  attention_mask.shape 是inputs的shape的前n-1个维度
            mask = torch.ones(inputs.shape[: -1], dtype=inputs.dtype).to(inputs.device)

        #  [batch_size, 1, 1, sequence_length]
        unsqu_mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 1)

        # 设置attention的填充部分(attention中为0的部分)为-10000，在后面softmax时，-10000的值对应的softnmax值接近0了
        unsqu_mask = (1.0 - unsqu_mask) * -10000.0

        # 应用attention
        energy = energy + unsqu_mask

        # 应用softmax，转换为softmax概率
        attention_scores = torch.softmax(energy, dim=-1) # [batch_size, num_heads, seq_len, seq_len]不变

        # [batch_size, num_heads, seq_len, seq_len] * [batch_size, num_heads, seq_len, head_dim]
        # = [batch_size, num_heads, seq_len, head_dim]
        context_layer = torch.matmul(self.dropout(attention_scores), V) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [batch_size, seq_len, num_heads, head_dim]

        # 合并后面两个维度
        context_layer = context_layer.reshape(batch_size, -1, self.embedding_dim)

        context_layer = self.layer_norm(context_layer + residual)
        context_layer = self.out_fc(context_layer)

        #  [batch_size, sequence_length, 1]
        mask = mask.unsqueeze(dim=2)
        context_layer = torch.mul(context_layer, mask)

        if need_weights:
            return context_layer, attention_scores.sum(dim=1) / self.num_heads
        
        return context_layer, None

class HierarchicalAttention(nn.Module):
    """
    层次注意力
    参考公式：https://zhuanlan.zhihu.com/p/26892711
    u_it = tanh(W_w * h_it, b_w)
    alph_it = exp(u_it * u_w) / sum(exp(u_it * u_w))
    score = sum(alph_it * h_it)
    """
    def __init__(self, embedding_dim: int):
        '''
        embedding_dim: 输出的隐藏层的维度
        '''
        super(HierarchicalAttention, self).__init__()

        # u_it = tanh(w_w * h_it + b_w)
        self.contribute_layer = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.Tanh(),
        )

        self.attention_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
            bias=False,     #不需要偏置值
        )

    def forward(self, inputs: torch.Tensor, return_scores: bool=False):
        '''
        inputs: [batch_size, max_seq_len, embedding_dim]

        return: 
            return_score=True: [batch_size, seq_len 1], 返回seq_len中每个t的权重
            return_score=False: [batch_size, hidden_szie]，返回原始向量和分数点乘、求和后的结果
        
        '''
        # u_it = tanh(W_w * h_it, b_w)
        u_it = self.contribute_layer(inputs)     # [batch_size, seq_len, embedding_dim]

        # [bathc_size, seq_len, 1]
        attention = self.attention_layer(u_it)  

        # alph_it = exp(u_it * u_w) / sum(exp(u_it * u_w))
        scores = F.softmax(attention, dim=1)    # [bathc_size, seq_len, 1]

        if return_scores:
            return scores

        # score = sum(alph_it * h_it)
        outputs = torch.mul(inputs, scores)     # [batch_size, seq_len, hidden_szie]
        outputs = torch.sum(outputs, dim=1)     # [bathc_size, hidden_szie]
        
        return outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_prob: float=0.0, device: str='cuda'):
        '''
        多头注意力
        '''
        super(MultiHeadAttention, self).__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError('embedding_dim must be positive integer mutiple of num_heads.')

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = int(embedding_dim / num_heads)

        self.query_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value_layer = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm((embedding_dim))

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def transpose_tensor(self, x: Tensor, batch_size: int):
        '''
        x: [batch_size, seq_len, embedding_dim]
        out: [batch_size, num_heads, seq_len, head_dim]
        '''
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)    # -1是指seq_len
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_mask: Tensor=None, key_mask:Tensor=None, need_weights: bool=False):
        '''
        query: [batch_size, query_len, embedding_dim]
        key: [batch_size, key_len, embedding_dim]
        value: [batch_size, value_len, embedding_dim]

        query_mask:
            [batch_size, query_len]

        key_mask: 
            if self attention (query = key = value):
                mask = [batch_size, 1, seq_len, seq_len]
            if not self attention (query != key and query != value):
                mask = [batch_size, key_len]
        '''
        batch_size = query.shape[0]
        residual = query 

        # 线性变换，输入输出的维度不变
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        # [batch_size, num_heads, seq_len, head_dim]
        Q = self.transpose_tensor(Q, batch_size)
        K = self.transpose_tensor(K, batch_size)
        V = self.transpose_tensor(V, batch_size)

        # [batch_size, num_heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if key_mask is not None:
            # [batch_size, 1, 1, key_len]
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(key_mask == 0.0, -10000.0)

        # [batch_size, num_heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)

        # [batch_size, num_heads, query_len, head_dim]
        context_layer = torch.matmul(self.dropout(attention), V)
        
        # [batch_size, query_len, num_heads, head_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # [batch_size, query_len, embedding_dime]
        context_layer = context_layer.reshape(batch_size, -1, self.embedding_dim)
        context_layer = self.layer_norm(context_layer + residual)

        # [batch_size, query_len, embedding_dime]
        context_layer = self.output_fc(context_layer)

        # [batch_size, query_len, 1]
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(dim=2)
            context_layer = torch.mul(context_layer, query_mask)

        if need_weights:
            return context_layer, attention.sum(dim=1) / self.num_heads
        
        return context_layer, None


if __name__ == "__main__":
    
    test_multiAtt = True
    test_h_att = False
    test_selfAtt = False

    if test_multiAtt:
        q = torch.randn((2, 3, 4))
        # k.shape == v.shape
        k = torch.randn((2, 4, 4))
        v = torch.rand((2, 4, 4))
        query_mask = torch.FloatTensor([[1,1,0],[1,1,1]])
        key_mask = torch.FloatTensor([[1,1,0,0],[1,1,1,1]])

        m_att = MultiHeadAttention(embedding_dim=4, num_heads=2, device='cpu')

        context, weights = m_att(query=q, key=k, value=v, query_mask=query_mask, key_mask=key_mask, need_weights=True)
        print(context.shape)
        print(weights.shape)
        print('=======')
        print(context)
        print(weights)

    if test_h_att:    
        h_attention = HierarchicalAttention(embedding_dim = 4)
        inputs = torch.randn((32,8, 12, 4))
        out = h_attention(inputs)
        print(out.shape)

    if test_selfAtt:
        attention = SelfAttention(embedding_dim=4, num_heads=2, device='cpu')
        x = torch.randn((2, 3, 4))
        mask = torch.FloatTensor([[1,1,0], [1,1,1]])
        out, w = attention(x, mask, need_weights=True)
        print(out.shape)
        print(w.shape)
        print(out)
        print(w)
    