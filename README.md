# 文本情感分类

## 任务简介
文本情感分类任务针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极、中性。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控。

先进行数据预处理和单词表生成，然后分别建立cnn、gru、bow、lstm、bilstm 4个文本情感分类模型，再用训练好的模型对测试集中的文本情感进行预测，判断其情感为「Negative」或者「Positive」。结果按照AUC（Area Under Curve）评估指标进行评估。

该实现参考自百度开源系统[Senta](https://github.com/baidu/Senta)。

## 构建准备
### 环境要求
Paddlepaddle v1.4.0 </br>
Python v3.6.2 </br>
sklearn </br>
pandas </br>
numpy </br>
tqdm

### 数据准备
数据格式如下：

NO |  列名  |   类型  | 字段描述
:-:|:-:|:-:|:-
1  |  ID    |  int   |文本唯一标识
2  | review | string | 文本记录
3  | label  | string | 文本的情感状态
   
以下是两条示例数据：

```
1,Jo bhi ap se tou behtar hoon,Negative
2,ya Allah meri sister Affia ki madad farma,Positive
...
```
详见data/train.csv。


## 模型训练与预测

### 构建词典

统计出现的词语构建词典，供模型训练使用，词典的格式为：每行一个词典项。

``` bash
python get_vocab.py > ./kesci_data/train.vocab
```
### 训练/评估/预测

``` bash
rm -r -f ./kesci_model/*
python sentiment_classify.py \
        --train_data_path ./kesci_data/train_data.csv \             # train_data path
        --test_data_path ./kesci_data/validation_data.csv \         # test_data path
        --word_dict_path ./data/train.vocab \                       # word_dict path
        --result_path ./kesci_data/result.csv \                     # result data path
        --mode train \                                              # train/eval/infer mode
        --model_path ./kesci_model                                  # model path (save/pred) 
```

**参考文献**

[1] APA Zhang, L. , Wang, S. , & Liu, B. . (2018). Deep learning for sentiment analysis: a survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, e1253.

    
