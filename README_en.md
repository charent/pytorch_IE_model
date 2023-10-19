# A information extract model base on pytorch

* *中文请点这里：[中文](README.md)*
Extract all triples in a piece of text, such as the sentence `李庆绿李庆绿，斋号枕涛居，1973年生，广东揭西人`, extract the triples `('李庆绿', '出生日期', '1973年')` and `('李庆绿', '出生地', '广东揭西')`

This project designed two hierarchical binary annotation information extraction models based on the attention mechanism: the `P-SO` model and the `SP-O` model. On the basis of shared coding, the attention mechanism is used to fuse the flow between the two subtasks. information, making the two subtasks more closely related. The two models designed in this project have simple feature engineering and do not use any natural language processing tools, such as word segmentation, part-of-speech tagging, etc., while avoiding the introduction of new errors, and at the same time, the inference and decoding speed in engineering applications is faster.

Under the simple feature engineering of the information extraction model using only word vectors and position vectors, the `F1` score of the `SP-O` model reaches `0.801`. In order to solve the problem of slow decoding one by one, this project designed a batch inference decoding method. Using the GTX 1050Ti graphics card, the inference decoding speed on the data set reached 359 items/second. Compared with the single decoding speed, the average increase was 817%. .
## Python Environment：
* python >= 3.7，3.7、3.8、3.10 has been tested.
* cuda >= 10.1，suggest 10.2.

# Install dependencies：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

## Download pretrained wordvec model（Optional, not using pretained from default）:
- Chinese Bert、Albert's pytorch version download url： 
- Bert：[bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main)
- Albert：[albert-base-chinese-cluecorpussmall](https://huggingface.co/uer/albert-base-chinese-cluecorpussmall)

- move downloaded files to folder model_file，and change variable in "*embeddeing.py" file of embedding folder（TORCH_BERT_DIR、ALBERT_BERT_DIR）.
- word2vec pretrained wordver download: [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- if your memory (less 16GB for most cases) is not enough to load all of word2vec vertor and throw an exception OOM (out of memory), please run this command to extract needed words' vector.
```bash
python ./utils/pertrain_to_numpy.py
```

## Download dataset： 
- <https://aistudio.baidu.com/aistudio/datasetdetail/11384>
- the downloaded file of train_data.json and dev_data.json should be moving on "data" folder.
```bash
--/data
----train_data.json
----dev_data.json
```
  
## Run or Training:
```bash
# first to process date
python ./utils/process_raw_data.py

# change model's config
vi config.py

# Training, main.py's first args should be `train_sp_o`, `train_p_so`, `test_sp_o`, `test_p_so`
# it means：training sp_o model, training p_so model、testing sp_o model, testing p_so model
python main.py train_sp_o
    
```

# Cite
If you feel this project is helpful to you, please cite the following paper:
```conf
https://kns.cnki.net/kcms2/article/abstract?v=7P_nOixU6lV4c5lbZIlAA0L1xyOeieRVh40VHRxIaT47vFCr5BStyodVzKStzcqTifi863z2Nx9bIsaXKF9OlV_GYlkPV8zEsSoCoUhFVa2gmjrqNDhdzQrcaPKHn8lQA2FkYQuO0co=&uniplatform=NZKPT&flag=copy
```