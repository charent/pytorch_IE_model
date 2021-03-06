from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError
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

class Renmin_embedding:
    '''
    人民日报词向量，使用lmdb，不加载词向量到内存
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
        super().__init__()

        # 如果已经初始化了，就不再初始化
        if self._is_init:
            return     
        self._is_init = True

        if not embedding_path:
            embedding_path = parent_path + '/model_file/merge_sgns_bigram_char300'
        log.info('正在初始化lmdb词向量......')
        self.embedding = LmdbEmbeddingsReader(embedding_path)

    @staticmethod
    def write_embedding(input_file: str, output_path: str):
        """
        从文本文件中写入embedding
        :param input_file: 输入路径
        :param output_path: 输出路径
        """
        print('start writing embeddings')

        def iter_embeddings():
            fin = open(input_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            print('words number:{num},dimension:{dim}'.format(num=n, dim=d))
            num = 0
            skipped = 0
            for line in fin:
                num = num + 1
                if num % 10000 == 0:
                    print('processed', num, 'words')
                line = line.rstrip().split()
                try:
                    word_ele, wv = line[0], np.array(line[1:], dtype=np.float32)
                except Exception:
                    skipped = skipped + 1
                    continue
                yield word_ele, wv

            print('skipped', skipped, 'lines')

        print('Writing vectors to a LMDB database...')

        writer = LmdbEmbeddingsWriter(
            iter_embeddings()
        ).write(output_path)

    def get_embedding(self, word: str):
        try:
            vector = self.embedding.get_word_vector(word)
        except MissingWordError:
            vector = np.ones(300) * 1e-9
        return vector
    
    def get_words_embedding(self, texts: str):
        '''
        返回一个句子的所有字向量
        '''
        vector = []
        get_word_vector = self.embedding.get_word_vector

        for word in texts:
            word_vec = None
            try:
                word_vec = get_word_vector(word)
            except MissingWordError:
                word_vec = np.ones(300) * 1e-8

            vector.append(word_vec)

        return np.array(vector)

    def get_sentence_embedding(self, sentence: str):
        if not sentence:
            return np.ones(300) * 1e-8
            
        return np.mean([self.get_embedding(word) for word in sentence], axis=0)

if __name__ == "__main__":
    embedding = RenminEmbedding()
    print(embedding.get_words_embedding('你好').shape)