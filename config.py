from dataclasses import dataclass
import platform

@dataclass
class Config(object):
    epoch = 100
    batch_size = 32 if platform.system() == 'Windows' else 64
    learning_rate = 0.001

    # 生成批处理数据时的进程数量
    num_workers = 0 if platform.system() == 'Windows' else 1

    # 是否将loss保存到文件
    log_loss = False

    # 最后一个epoch的学习率衰减为初始学习率的 1 / 10 （大约）
    lr_T_max = int(epoch * 0.1)

    embedding_size = 256
    cuda_device_number = 0

    # 训练时，用前多少个epoch做warm up
    warm_up_epoch = 1

    # output linear forward dim
    forward_dim = int(embedding_size * 2)

    # rnn_type = ['gru', 'lstm']
    rnn_type = 'gru'

     # gru or lstm hidden_size
    rnn_hidden_size = 256
    
    # MutilHeadAttention / SelfAtttention heads
    # 注意力头数必须能被词向量维度整除，embedding_size % num_heads === 0
    num_heads = 8

    assert embedding_size % num_heads == 0

    # 预训练词向量，选项为['none','word2vec', 'albert', 'bert']
    # 对应的词向量为: [embedding_size, 300, 768, 768]
    from_pertrained = 'none'

    # bert
    bert_forward_dim = 256
    # 'cpu' or 'cuda'/'cuda:0'
    bert_device = 'cuda:0'

    # legacy:
    predicate_embedding_szie = 64
    sigmoid_threshold = 0.5