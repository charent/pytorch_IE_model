from gensim.models import KeyedVectors
from os.path import dirname, abspath
import sys
import numpy as np
import threading

sys.path.append('..')
sys.path.append('.')
from utils.logger import Logger

log = Logger('renmin_embedding').get_logger()
parent_path = abspath(dirname(dirname(__file__)))

class RenminEmbedding:
    '''
    人民日报词向量，把所有的词向量加载到内存
    '''
    _instance_lock = threading.Lock()
    _is_init = False

     # 单例模式
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with cls._instance_lock:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, embedding_path=None):
        '''
        embedding_path下要有两个文件：
            merge_sgns_bigram_char300.bin
            merge_sgns_bigram_char300.bin.vectors.npy
        缺一不可
        '''
        super().__init__()

        # 如果已经初始化了，就不再初始化
        if self._is_init:
            return
        self._is_init = True

        if not embedding_path:
            embedding_path = parent_path + '/model_file/merge_sgns_bigram_char300'
        log.info('正在加载词向量......')
        self.embedding = KeyedVectors.load(embedding_path + '/merge_sgns_bigram_char300.bin')

    def get_words_embedding(self, texts: str):
        '''
        返回一个句子的所有字向量
        '''
        vector = []
        embedding = self.embedding

        for word in texts:
            word_vec = None
            if word in embedding:
                word_vec = embedding[word]
            else:
                word_vec = np.ones(300) * 1e-9
            vector.append(word_vec)

        return np.array(vector)

if __name__ == "__main__":
    embedding = RenminEmbedding()
    print(embedding.get_words_embedding('你好').shape)