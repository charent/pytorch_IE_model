import codecs
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import ujson
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PertrainEmbedding2numpy(object):

    def __init__(self, chars_file: str, pertrain_embedding_path: str):
        
        self.chars = []
        append = self.chars.append
        with codecs.open(chars_file, encoding='utf-8') as f:
            char2id = ujson.load(f)
            for char, cid in char2id.items():
                append(char)

        print('loadiing pertrain embedding ...')
        self. pertrain_embedding = KeyedVectors.load(pertrain_embedding_path)


    def convert2numpy(self, save_numpy_file: str, embedding_szie: int):
        print('convert ...')

        chars_len = len(self.chars)
        pertrain_embedding = self.pertrain_embedding

        # 随机初始化词向量矩阵，如果不存在该词，就用随机的向量代替
        # chars中已经添加一个填充字符pad, 一个未知字符unk
        embedding_init = np.random.normal(loc=-1.0, scale=1.0, size=(chars_len, embedding_szie))

        find_char_count = 0

        for index, char in tqdm(enumerate(self.chars)):
            if char in pertrain_embedding:
                #  索引还是从0开始，
                embedding_init[index] = pertrain_embedding[char]
                find_char_count += 1

        print('命中率: {:.4f} %'.format((find_char_count / chars_len) * 100))

        # 第0个词向量是[pad]，设置为0
        embedding_init[0] = np.zeros(shape=(embedding_szie, ), dtype=np.float)

        np.save(save_numpy_file, embedding_init)


if __name__ == "__main__":
    pertrain_embedding_path = BASE_DIR +  '/model_file/merge_sgns_bigram_char300/merge_sgns_bigram_char300.bin'
    chars_file = BASE_DIR + '/data/char2id.json'

    save_base = BASE_DIR + '/model_file/pertrain'

    if not os.path.exists(save_base):
        os.mkdir(save_base)
    save_numpy_file = save_base + '/merge_sgns_bigram_char300.npy'

    pertrain2numpy = PertrainEmbedding2numpy(chars_file=chars_file, pertrain_embedding_path=pertrain_embedding_path)
    pertrain2numpy.convert2numpy(save_numpy_file=save_numpy_file, embedding_szie=300)