import ujson
from tqdm import tqdm
import codecs
from os.path import dirname, abspath
import sys
from matplotlib import pyplot as plt
import re
import numpy as np

sys.path.append('.')
sys.path.append('..')
from utils.logger import Logger
from utils.function import repair_song_album

log = Logger('prepare_data').get_logger()

test = False

parent_path = abspath(dirname(dirname(__file__))) + '/data/'
TRAIN_FILE = parent_path + 'train_data.json'
DEV_FILE = parent_path + 'dev_data.json'

np.random.seed(23333)
DEV_SIZE = 2000

if test:
    TRAIN_FILE = parent_path + 'train_data_sample.json'
    DEV_FILE = parent_path + 'dev_data_sample.json'

ALL_50_SCHEMAS_FILE = parent_path + 'all_50_schemas'


def read_all_lines(path: str):
    log.info('读取：{}'.format(path))
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def process_all_50_schemas(path: str):
    log.info('处理关系映射......')

    # 第0种关系不使用，表示没有这种关系
    id2predicate = dict()
    predicate2id = dict()

    all_50_schemas = dict()
    all_predicate = set()
    lines = read_all_lines(path)

    object_type_replace = {
        'Number': '数字',
        'Text': '文本',
        'Date': '日期',
    }
    
    for line in lines:
        tmp = ujson.decode(line)

        object_type = tmp['object_type']
        if object_type in object_type_replace:
            object_type = object_type_replace[object_type]

        all_50_schemas[tmp['predicate']] = {'s_type': tmp['subject_type'], 'p_type': tmp['predicate'], 'o_type': object_type}
        all_predicate.add(tmp['predicate'])
    
    # 按照关系的长度排序
    all_predicate = list(all_predicate)
    all_predicate.sort(key=lambda x: len(x))

    all_predicate.insert(0, '[UNK]')

    for idx, predicate in enumerate(all_predicate):
        id2predicate[idx] = predicate
        predicate2id[predicate] = idx
    
    with codecs.open(parent_path + 'id_and_predicate.json', 'w', encoding='utf-8') as f:
        ujson.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

    with codecs.open(parent_path + 'predicate_info.json', 'w', encoding='utf-8') as f:
        ujson.dump(all_50_schemas, f, indent=4, ensure_ascii=False) 

def count_len(data: list):
    '''
    统计文本长度
    '''
    len_dict = {}
    
    for i in range(len(data)):
        sentence_len = len(data[i]['text'])
        
        if sentence_len in len_dict:
            len_dict[sentence_len] += 1
        else:
            len_dict[sentence_len] = 1
    
    lengths = [key for key in len_dict.keys()]
    lengths.sort()
    
    count = [len_dict[key] for key in lengths]
    
    texts_item_count = sum(count)

    # 统计前百分之多少的文本占比
    def count_per(per: float):
        sum_ = 0
        for i in range(len(lengths)):
            sum_ += count[i]

            if sum_ / texts_item_count >= per:
                return lengths[i], sum_

    len_, sum_ = count_per(0.8)
    log.info('80%的文本长度<=：{}, 共 {} 条'.format(len_, sum_))
    len_, sum_ = count_per(0.9)
    log.info('90%的文本长度<=：{}, 共 {} 条'.format(len_, sum_))
    len_, sum_ = count_per(0.95)
    log.info('95%的文本长度<=：{}, 共 {} 条'.format(len_, sum_))
    len_, sum_ = count_per(0.99)
    log.info('99%的文本长度<=：{}, 共 {} 条'.format(len_, sum_))
    len_, sum_ = count_per(1.0)
    log.info('100%的文本长度<=：{}, 共 {} 条'.format(len_, sum_))
    
    # plt.bar(lengths, count)
    # plt.show()

def process_spo_list(text: str, spo_list: list, repair_song: bool=False):
    '''
    处理spo_list,处理成{subject: 'subject', subject_start: 0, subject_end:3, predicate: 'predicate', object: 'object', object_start: 5, object_end = 7}
    '''
    new_spo_list = []

    # 找出所有用书名号隔开的名字
    some_name = re.findall('《([^《》]*?)》', text)
    some_name = [n.strip() for n in some_name]
    
    # 歌曲和专辑
    song = []
    album = []
    for spo in spo_list:
        temp = dict()

        # 修正so的错误，删除前后的书名号
        s = spo['subject'].strip('《》').strip().lower()
        o = spo['object'].strip('《》').strip().lower()
        p = spo['predicate']
        
        # 如果s在找到的名字中，以正则找到的s为准，用in判等，
        # 如text: '《造梦者---dreamer》'，但是标注的s是'造梦者'
        for name in some_name:
            if s in name and text.count(s) == 1:
                s = name
        
        if repair_song:
            if p == '所属专辑':
                song.append(s)
                album.append(o)

        temp = dict()
        temp['subject'] = s
        temp['subject_start'] = text.find(s)
        temp['subject_end'] = temp['subject_start'] + len(s)
        temp['predicate'] = spo['predicate']
        temp['object'] = o
        temp['object_start'] = text.find(o)
        temp['object_end'] = temp['object_start'] + len(o)

        # 在text中找不到subject获取object，不要这条数据了
        if temp['subject_start'] == -1 or temp['object_start'] == -1 or  len(temp) == 0:
            continue

        new_spo_list.append(temp)
    
    if repair_song:
        ret_spo_list = []
        ps = ['歌手', '作词', '作曲']
        
        for spo in new_spo_list:
            s, p, o = spo['subject'], spo['predicate'], spo['object']
            if p in ps and s in album and s not in song:
                continue
            ret_spo_list.append(spo)

        return ret_spo_list

    return new_spo_list


def process_data(path: str, save_file_name: str, all_chars: set, dev_file_name: str=None, keep_max_length: int=300, repair_song: bool=False):
    lines = read_all_lines(path)
    my_raw_data = []

    log.info('处理前的总行数:{}'.format( len(lines) ))

    for i, line in tqdm(enumerate(lines)):
        
        tmp = ujson.decode(line)
        text = tmp['text'].lower()

        for char in text:
            char = char.strip()
            if len(char) > 0:
                all_chars.add(char)
            
        spo_list = process_spo_list(text, tmp['spo_list'], repair_song=repair_song)

        # 删除长度过长、没有找到实体信息的句子
        if len(tmp['text']) > keep_max_length or len(spo_list) == 0:
            continue

        my_raw_data.append({'text': text, 'spo_list': spo_list})

    log.info('处理后的总行数:{}，丢掉{}条在句子中找不到实体、长度太长数据'.format(len(my_raw_data),  len(lines) - len(my_raw_data) ))
    count_len(my_raw_data)

    if dev_file_name is not None:
        dev_index = np.random.choice(range(0, len(my_raw_data)), size=DEV_SIZE, replace=False)
        dev_index = set(dev_index)
        assert len(dev_index) == DEV_SIZE
        
        train_data = [x for i, x in enumerate(my_raw_data) if i not in dev_index]
        dev_date = [x for i, x in enumerate(my_raw_data) if i in dev_index]
        
        with codecs.open(parent_path + dev_file_name, 'w', encoding='utf-8') as f:
            ujson.dump(dev_date, f, indent=4, ensure_ascii=False)
        
        my_raw_data = train_data

    with codecs.open(parent_path + save_file_name, 'w', encoding='utf-8') as f:
        ujson.dump(my_raw_data, f, indent=4, ensure_ascii=False)

   
def merge_all_chars():
    '''
    合并两个词库
    '''
    all_chars = set()
    with codecs.open(parent_path + 'all_chars.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with codecs.open(parent_path + 'vocab.txt', 'r', encoding='utf-8') as f:
        lines.extend(f.readlines())
    
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            all_chars.add(line)

    all_chars = list(all_chars)
    all_chars.sort()

    with codecs.open(parent_path + 'vocab.txt', 'w', encoding='utf-8') as f:
       f.writelines([char + '\n' for char in all_chars if not char.endswith('\n')])

if __name__ == "__main__":
    process_all_50_schemas(ALL_50_SCHEMAS_FILE)
    all_chars = set()

    max_seq_len = 200
    repair_song = False
    dev_file_name='my_dev_data.json'

    process_data(TRAIN_FILE, 'my_train_data.json', dev_file_name=dev_file_name, all_chars=all_chars, \
        keep_max_length=max_seq_len, repair_song=repair_song)
    process_data(DEV_FILE, 'my_test_data.json', all_chars, keep_max_length=max_seq_len, repair_song=repair_song)
  
    all_chars = list(all_chars)
    all_chars.sort()

    char2id = dict()
    char2id['[PAD]'] = 0
    char2id['[UNK]'] = 1
    for idx, char in enumerate(all_chars):
        char2id[char] = idx + 2

    with codecs.open(parent_path + 'char2id.json', 'w', encoding='utf-8') as f:
        ujson.dump(char2id, f, indent=4, ensure_ascii=False) 

    # merge_all_chars()

    # # test:
    # process_data(BASE_DIR + 'train_data_sample.json', 'my_train_data.json')
    # process_data(BASE_DIR + 'dev_data_sample.json', 'my_dev_data.json')

