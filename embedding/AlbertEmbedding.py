from transformers import AlbertModel
from os.path import dirname, abspath
import sys
import torch.nn as nn
from typing import Union

sys.path.append('.')
sys.path.append('..')

from embedding.bert_embedding import BertTokenizerCN

parent_path = abspath(dirname(dirname(__file__)))
ALBERT_BERT_DIR = parent_path + '/model_file/albert_base_zh'
VOCAB = ALBERT_BERT_DIR + '/vocab.txt'

class AlbertEmbedding(nn.Module):
    """
    加载训练好的模型使用
    """
    def __init__(self, vocab_file: str=VOCAB, torch_bert_path: str=ALBERT_BERT_DIR, device: str='cpu'):
        super(AlbertEmbedding, self).__init__()
      
        print('albert device: {}'.format(device))
        
        self.device = device

        # 加载tokenizer
        self.tokenizer = BertTokenizerCN(vocab_file=vocab_file, torch_bert_path=torch_bert_path)

        # 加载bert模型
        self.model = AlbertModel.from_pretrained(torch_bert_path).to(device)
        self.model.eval()

    def text_to_embedding(self, text: Union[list, str], max_len: int=None,  requires_grad: bool=False):
        '''
        return: 
            input_embedding, cls_embedding, length, attention_mask
        '''
        return self.forward(text, max_len, requires_grad)

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

if __name__ == "__main__":
    device = 'cuda'
    bert_embedding = AlbertEmbedding(device=device)
    text = ['你好', 'albert模型']
    input_embedding, cls_embedding, length, attention_mask = bert_embedding(text)
    # print(input_embedding[0])
    print(input_embedding.shape)
    print(cls_embedding.shape)
    print(length)
    print(attention_mask)