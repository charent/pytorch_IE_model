# pytorch_information_extract_model
# 基于pytorch的信息抽取模型

*Read this in [English](README_en.md).*

## 环境要求：
* python >= 3.7，3.7、3.8、3.10测试无问题
* cuda >= 10.1，建议10.2

## 安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

## 下载预训练模型（可选，默认不使用预训练模型）:
- 中文Bert、Albert的pytorch版本下载地址： 
- Bert：<https://huggingface.co/bert-base-chinese/tree/main>
- Albert：<https://huggingface.co/uer/albert-base-chinese-cluecorpussmall> 

- 下载的文件存放搭配model_file文件夹下，并修改embedding下对应的 "*embeddeing.py"文件开头的中定义的目录（TORCH_BERT_DIR、ALBERT_BERT_DIR）。
- word2vec预训练词向量下载地址：<https://github.com/Embedding/Chinese-Word-Vectors>
- 下载的word2vec预训练可能太大导致内存不足，运行下面的代码进行处理（提取出数据集中包含的字向量，其他的词向量不要）
```bash
python ./utils/pertrain_to_numpy.py
```

## 数据集下载： 
- <https://aistudio.baidu.com/aistudio/datasetdetail/11384>
- 下载的数据集dev_data.json放到data文件夹下
```bash
--/data
----dev_data.json
```
  
## 运行:
```bash
# 先处理数据
python ./utils/process_raw_data.py

# 修改config配置
vi config.py

# 修改main.py确定运行哪个框架
vi main.py

# 运行, main.py的第一个参数可以是`train_sp_o`、`train_p_so`、`test_sp_o`、`test_p_so`
# 分别表示：训练sp_o模型、训练p_so模型、测试sp_o模型、测试p_so模型
python main.py train_sp_o
    
```