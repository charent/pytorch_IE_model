# pytorch_information_extract_model
# A information extract model base on pytorch

* 中文请点这里：[中文](README.md)*

## Python Environment：
* python >= 3.7，3.7、3.8、3.10 has been tested.
* cuda >= 10.1，suggest 10.2.

# Install dependencies：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host tuna.tsinghua.edu.cn
```

## Download pretrained wordvec model（Optional, not using pretained from default）:
- Chinese Bert、Albert's pytorch version download url： 
- Bert：<https://huggingface.co/bert-base-chinese/tree/main>
- Albert：<https://huggingface.co/uer/albert-base-chinese-cluecorpussmall> 

- move downloaded files to folder model_file，and change variable in "*embeddeing.py" file of embedding folder（TORCH_BERT_DIR、ALBERT_BERT_DIR）.
- word2vec pretrained wordver download url：<https://github.com/Embedding/Chinese-Word-Vectors>
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