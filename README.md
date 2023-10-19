# 基于二分标注的信息抽取模型

*Read this in [English](README_en.md).*

抽取出一段文本中所有的三元组，如句子`李庆绿李庆绿，斋号枕涛居，1973年生，广东揭西人`，抽取出三元组`('李庆绿', '出生日期', '1973年')`和`('李庆绿', '出生地', '广东揭西')`

本项目设计了两个基于注意力机制的层次二分标注信息抽取模型：`P-SO`模型和`SP-O`模型，在共享编码的基础上，利用注意力机制融合两个子任务之间流动信息，使得两个子任务关联性更强。本项目设计的两个模型特征工程简单，没有利用任何自然语言处理工具，如分词、词性标注等，避免引入新误差的同时，在工程应用中的推断解码速度更快。 

信息抽取模型在只利用字向量和位置向量的简单特征工程下，`SP-O`模型的`F1`分数达到`0.801`。为解决逐条解码慢的问题，本项目设计了批处理推断解码方法，用`GTX 1050Ti`显卡在数据集上的推断解码速度达到`359条/秒`，对比单条解码速度平均提升`817%`。

## 环境要求：
* python >= 3.7，3.7、3.8、3.10测试无问题
* cuda >= 10.1，建议10.2

## 安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

## 下载预训练模型（可选，默认不使用预训练模型）:
- 中文Bert、Albert的pytorch版本下载地址： 
- Bert：[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)
- Albert：[albert-base-chinese-cluecorpussmall](https://huggingface.co/uer/albert-base-chinese-cluecorpussmall)

- 下载的文件存放搭配model_file文件夹下，并修改embedding下对应的   `embeddeing.py`文件开头的中定义的目录（`TORCH_BERT_DIR`、`ALBERT_BERT_DIR`）。
- word2vec预训练词向量下载地址：[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
  
- 下载的word2vec预训练可能太大导致内存不足，运行下面的代码进行处理（提取出数据集中包含的字向量，其他的词向量不要）
```bash
python ./utils/pertrain_to_numpy.py
```

## 数据集下载： 
- <https://aistudio.baidu.com/aistudio/datasetdetail/11384>
- 下载的数据集train_data.json、dev_data.json放到data文件夹下
```bash
--/data
----train_data.json
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

# 引用
如果你觉得本项目对你有所帮助，请引用以下论文：
```conf
https://kns.cnki.net/kcms2/article/abstract?v=7P_nOixU6lV4c5lbZIlAA0L1xyOeieRVh40VHRxIaT47vFCr5BStyodVzKStzcqTifi863z2Nx9bIsaXKF9OlV_GYlkPV8zEsSoCoUhFVa2gmjrqNDhdzQrcaPKHn8lQA2FkYQuO0co=&uniplatform=NZKPT&flag=copy
```