# from pytorch_transformers import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel
from transformers.tokenization_utils import _is_whitespace
import collections
from os.path import dirname, abspath
import torch
import torch.nn as nn
import codecs
from typing import Union

parent_path = abspath(dirname(dirname(__file__)))

# Bert模型文件路径，请根据实际情况修改
TORCH_BERT_DIR = parent_path + '/model_file/bert'

VOCAB = TORCH_BERT_DIR + '/vocab.txt'

class Bert_embedding(nn.Module):
    """
    加载训练好的bert模型使用
    """
    def __init__(self, device: str='cpu'):
        super().__init__()
        print('bert device: {}'.format(device))
        self.device = device

        # 加载tokenizer
        self.tokenizer = BertTokenizerCN(vocab_file=VOCAB, torch_bert_path=TORCH_BERT_DIR)

        # 加载bert模型
        self.model = BertModel.from_pretrained(TORCH_BERT_DIR).to(device)
        self.model.eval()

    def text_to_embedding(self, text: Union[list, str], max_len: int=None):
        '''
        return: 
            input_embedding, cls_embedding, length, attention_mask
        '''
        return self.forward(text, max_len)

    def forward(self, text: Union[list, str], max_len: int=None, requires_grad: bool=False):
        if isinstance(text, str):
            text = [text]

        if max_len is None:
            max_len = max([len(x) for x in text]) + 2   # 2 个特殊token： cls, sep
        device = self.device

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, return_length=True, pad_to_multiple_of=max_len)
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        length = inputs['length']

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        input_embedding = outputs[0]
        cls_embedding = outputs[1]

        if not requires_grad:
            input_embedding = input_embedding.detach()
            cls_embedding = cls_embedding.detach()
            
        return input_embedding.to(device), cls_embedding.to(device), length.to(device), attention_mask.to(device)

# 继承BertTokenizer，重写_tokenize方法，保证句子按照字符划分
# 否则序列标注会因为去掉空格、按照词划分而产生错误
class BertTokenizerCN(BertTokenizer):
    def __init__(self, vocab_file: str=VOCAB, torch_bert_path: str=TORCH_BERT_DIR):
        # 父类用from_pretrained方法构造
        super().__init__(vocab_file=vocab_file)

        # 参照BertTokenizer读取字典的方法
        token_dict = collections.OrderedDict()
        with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for index, char in enumerate(lines):
                char = char.rstrip("\n")
                token_dict[char] = index
        
        self.token_dict = token_dict

    def _tokenize(self, text: str):
        '''
        重写_tokenize方法，只按字划分，不按词划分,空白字符用'[unused1]'代替
        input: 'abc 你 好'
        rturn: ['a','b','c','你', '[unused1]','好']
        '''
        ret = []
        append = ret.append
        token_dict = self.token_dict
        for char in text:
            if char in token_dict:
                append(char)
            elif _is_whitespace(char):
                append('[unused1]')     # 用vocab字典中的未训练字符'[unused1]'代替空白字符
            else:
                append('[UNK]')     # 不能识别的字符
        
        return ret


if __name__ == "__main__":

    device = 'cuda'
    bert_embedding = Bert_embedding(device)
    text = ['你好', 'bert模型']
    input_embedding, cls_embedding, length, attention_mask = bert_embedding(text)
    # print(input_embedding[0])
    print(input_embedding.shape)
    print(cls_embedding.shape)
    print(length)
    print(attention_mask)
   