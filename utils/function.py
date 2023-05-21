import time
from numpy.lib.function_base import append
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ModuleList, Module
import torch.nn.functional as F
import codecs
import ujson
import copy
import numpy as np

def read_json(path: str):
    '''
    加载json文件
    '''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        return ujson.load(f)

def tensor_cat_repeat_n(a: Tensor, b: Tensor):
    '''
    将b重复seq_len次，再拼接到a后面
    args:
        a.shape: [batch_size, seq_len, size_a]
        b.shape: [batch_size, size_b] 或者 [batch_size, 1, size_b]
    return:
        output: [batch_size, seq_len, size_a + size_b]
    '''
    seq_len = a.shape[1]

    if len(b.shape) == 2:
        b = torch.unsqueeze(b, dim=1)   # [batch_szie, 1, embedding_szie]

    b = torch.repeat_interleave(b, repeats=seq_len, dim=1)  # [batch_size, seq_len, embedding_size]
    output = torch.cat([a, b], dim=2)  # [batch_size, seq_len, embedding_size * 2]

    return output

def Ndim_tensor_repeat_n(reapet_tensor: Tensor, lengths: Tensor, max_seq_len: int):
    '''
    将不定长的reapet_tensor重复max_seq_len次，超出max_seq_len的部分被截断
    '''
    out_tensor = []

    for bs_id, length in enumerate(lengths):
        rep_tensor = reapet_tensor[bs_id, 0: length, :]    # [len, embedding_size]
        repeat_n = torch.ceil(max_seq_len / length.float()).long()    # 要重复多少次才能达到max_seq_len, 向上取整

        rep_tensor = torch.repeat_interleave(rep_tensor, repeats=repeat_n, dim=0)[0: max_seq_len, :]
        out_tensor.append(torch.unsqueeze(rep_tensor, dim=0))
    
    # [batch_size, max_seq_len, embedding_size]
    out_tensor = torch.cat(out_tensor, dim=0)
    
    return out_tensor

def create_mask_from_lengths(lengths: Tensor, max_len: int=None):
    '''
    通过lengths数组创建mask
    输入：[1,2,3],max_len = 5
    输出：[
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,1,1,0,0]
    ]
    '''
    if max_len is None:
        max_len = torch.max(lengths)
    device = lengths.device
    mask = torch.arange(max_len).expand(len(lengths), max_len).to(device) < lengths.unsqueeze(1)
    mask = mask.float().detach()
    return mask

def f1_p_r_compute(spo_list_pred: list, spo_list_true: list, repair: bool=False):
    '''
    spo_list: [ [(s,p,o)...], [(s,p,o)]], 每一行[(s,p,o)...]为一个句子中的spo
    计算spo的f1分数，精确率，召回率，
    '''
    assert len(spo_list_pred) == len(spo_list_true)
    if repair:
        spo_list_pred = repair_song_album_list(spo_list_pred)
        spo_list_true = repair_song_album_list(spo_list_true)

    TP = 1e-10      # 正类判定为正类, A
    # TN = 1e-10    # 负类判定为负类
    TP_FP = 1e-10   # 检索到的, A + B
    TP_FN = 1e-10   # 真正想要的，A + C
    # FP = 1e-10    # 负类判定为正类
    # FN = 1e-10    # 正类判定为负类

    # p = a / (a + b)
    # r = a / (a + c)
    # f1 = 2pr / (p + r)

    for i in range(len(spo_list_true)):
        pred_set = set(spo_list_pred[i])
        true_set = set(spo_list_true[i])

        pred_true_set = pred_set & true_set     # 预测和真实取交集

        TP += len(pred_true_set)    # 检索到且是想要的， A
        TP_FP += len(pred_set)      # 检索到的，包括想要的和不想要的，A + B
        TP_FN += len(true_set)      # 真正想要的， 包括检索到和没检索到的，A + C

    p = TP / TP_FP
    r = TP / TP_FN
    f1 = (2 * p * r) / (p + r)
    
    return f1, p, r

def tensor_max_poll(seq: Tensor, mask: Tensor):
    '''
    对seq进行mask，然后做max pool
    seq:  [batch_size, seq_len, embed_dim]
    mask: [bath_size, seq_len]
    '''
    if len(mask.shape) == 2:
        mask = torch.unsqueeze(mask, dim=2)
    seq = seq - (1 - mask) * 1e9

    return torch.max(seq, dim=1)[0]

def tensor_avg_poll(seq: Tensor, mask: Tensor):
    '''
    对seq进行mask，然后做average pool
    seq:  [batch_size, seq_len, embed_dim]
    mask: [bath_size, seq_len]
    '''
    if len(mask.shape) == 2:
        mask = torch.unsqueeze(mask, dim=2)
    seq = seq * mask
    length = torch.sum(mask, dim=1)
    
    return torch.sum(seq, dim=1) / length

def get_formated_time():
    '''
    获取格式化的系统时间
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_models_parameters(model_list: list, weight_decay: float=0.0001):
    '''
    获取多个模型的可训练参数，包括模型内嵌套的网络参数
    多个模型放到一个list里面
    '''
    parameters = []
    no_decay = ["bias", "LayerNorm.weight"]

    for model in model_list:
        params = [{
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },{
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        parameters.extend(params)
    
    return parameters

def get_module_clones(module: Module, N: int):
    '''
    克隆多个torch网络
    '''
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def init_weights(m: Module):
    '''
    参数权重初始化,
    使用方法：
        model = Net()
        model.apply(init_weights)
    '''
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight.data)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0.0)
    # if isinstance(m, nn.Conv1d):
    #     nn.init.xavier_normal_(m.weight.data,)
    #     if m.bias is not None:
    #         m.bias.data.fill_(0.0)
    if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in n:
                nn.init.orthogonal_(p.data)
            elif 'bias' in n:
                p.data.fill_(0.0)

def print_and_log(msg: str, log):
    print(msg)
    log.info(msg)

def tensor_avg_pool1d(tensor: Tensor, kernel_size: int, mask: Tensor=None, stride: int=1):
    '''
    先对tensor做mask，再做1d averger poo，
    maske: [batch_size, seq_len]
    tensor: [batch_size, seq_len, embedding_dim]
    return: [batch_szie, seq_len, embedding_dim]
    '''
    if mask is not None:
        mask = mask.unsqueeze(2)
        tensor = tensor * mask
    ret = F.avg_pool1d(tensor, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2))

    return ret

def save_spo_list(dev_data: list, spo_list_pred: list, spo_list_true: list, file_path: str):
    '''
    '''
    assert len(dev_data) == len(spo_list_pred) == len(spo_list_true)

    with open(file_path, 'w', encoding='utf-8') as f:
        for data, spo_pred, spo_true in zip(dev_data, spo_list_pred, spo_list_true):
            T = set(spo_true)
            P = set(spo_pred)
            text = ujson.dumps({
                'text': data['text'],
                'spo_true': [spo for spo in spo_true],
                'spo_pred': [spo for spo in spo_pred],
                'spo_loss': [spo for spo in list(T - P)],
                'spo_new': [spo for spo in list(P - T)],
            }, ensure_ascii=False, indent=4)

            text += '\n'
            f.write(text)
    
def tensor_padding(t: Tensor, max_seq_len: int=256):
    '''
    对t进行填充，填充为最大长度：max_seq_len
    如果t的seq_len大于max_seq_len，则截断为max_seq_len
    t: 
        [batch_size, seq_len, embedding_dim]
    return:
        [batch_size, max_seq_len, embedding_dim]
    '''
    seq_len = t.shape[1]
    device = t.device

    if seq_len > max_seq_len:
        return t[:, 0: max_seq_len, :]

    batch_size = t.shape[0]
    embedding_dim = t.shape[2]
    pad_len = max_seq_len - seq_len

    pad_tensor = torch.zeros((batch_size, pad_len, embedding_dim)).to(device)
    ret = torch.cat([t, pad_tensor], dim=1)

    return ret

def tensor_gather(tensor: Tensor, start_end: np.array, max_n: int=8):
    '''
    从tensor中取出指定的开始位置到结束位置中的向量，自动对齐为max_n，大于max_n的会进行等间隔采样
    tensor:
        [batch_size, seq_len, embedding_dim]
    start_end:
        [batch_size, 2], 第一个数为开始位置，第二个数为结束位置，闭区间。
        如：[[0, 2],[1, 2]]，max_n=3，对齐的索引为[[0, 1, 2],[1, 2, 2]]
    return:
        [batch_size, max_n, embedding_dim]
    '''
    start = start_end[:, 0]
    end = start_end[:, 1]
    idxs = np.array([np.round(end * i + start * (1.0 - i)) for i in np.arange(max_n) / (max_n - 1.0)], dtype=np.int32)
    idxs = np.transpose(idxs)
    
    ret = []
    append = ret.append
    for ts, idx in zip(tensor, idxs):
        append(ts[idx, :])
      
    
    ret = torch.cat(ret, dim=0).reshape(-1, max_n, tensor.shape[2])
    return ret

def adjust_learning_rate(lr: float, optimizer: torch.optim):
    '''
    手动设置学习率
    '''
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def repair_song_album(spo_list: list, song: list, album: list):
    '''
    修复一条文本的'歌曲'和'专辑'的spo。
    对于歌曲x（subject）的关系歌手、作词、作曲，x必须同时存在于song和album中
    '''
    if len(song) == 0 and len(album) == 0:
        return spo_list

    ps = ['歌手', '作词', '作曲']
    new_spo_list = []
    for spo in spo_list:
        s, p = spo[0], spo[1]
        if p in ps and s in album and s not in song:
            continue
        new_spo_list.append(spo)
    
    return new_spo_list

def repair_song_album_list(spo_list: list):
    '''
    '''
    new_spo_list = []
    for spos in spo_list:
        song, album = [], []
        for spo in spos:
            s, p, o = spo
            if p == '所属专辑':
                song.append(s)
                album.append(o)
        new_spo_list.append(repair_song_album(spos, song, album))
    
    return new_spo_list