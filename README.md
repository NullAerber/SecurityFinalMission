# SecurityFinalMission

## 环境

> Python 环境： 3.6

## 说明
该项目是课程大作业，只具有实践复现效果，底层理论知识不提供。

是结合了[用机器学习玩转恶意URL检测](http://www.freebuf.com/articles/network/131279.html) 后的改进版本，
加入了[SVM支持向量机](http://www.freebuf.com/articles/web/130004.html)进行对比，
同时加入了以keras为框架的LSTM预测。

本项目相关具体介绍请[下载报告](https://github.com/NullAerber/SecurityFinalMission/paper.pdf)

## 数据集
本项目一共有三大类数据集，其中：

0 =正常请求日志样本

1 =请求日志表示尝试的注入攻击

bad_waf.txt,good_waf.txt:来自于[FWaf项目](https://github.com/faizann24/Fwaf-Machine-Learning-driven-Web-Application-Firewall)

bad1.txt, good1.txt,good2.txt:某系统的某天的正常访问url，已进行清洗去重

dev-access.csv:通过模拟API给出JSON形式的访问请求日志

## logical regression 和 SVM支持向量机
### 使用帮助
```
$ python BinaryClassification/train.py -h

Usage: train.py [options]

Options:
  -h, --help            show this help message and exit
  -c CLASSIFIER, --classifier=CLASSIFIER
                        classifier type：'lg' or 'svm'
  -g GOOD, --good=GOOD  good queries file
  -b BAD, --bad=BAD     bad queries file
  -n NGRAM, --ngram=NGRAM
                        the number of n gram
  -u USE, --use=USE     weather use kmeans
  -k KMEANS, --kmeans=KMEANS
                        the number of kmeanss
```
### 训练
```
$ python BinaryClassification/train.py [options]
```

可以设定相关参数
```
-c 分类器，是logical regression 还是SVM向量机
-g 选择postive样本的文本
-b 选择negative样本的文本
-n 分词时采用n长度的分词长度
-u 是否使用keams降维方法
-k 采用的话keams聚类到几类上
```

### 模型文件
.label文件保存的是分词结果，只与数据集有关

.pickle文件是训练后的模型

### 训练结果
训练均是默认训练集：

good = './BinaryClassification/data/good2.txt'

bad = './BinaryClassification/data/bad_waf.txt'

各种模式训练出来的结果：

type| logical regression| SVM
- | :-: | :-: 
K number (False)| 0.9987866676182999 | 0.9632431660837913 
K number (80) | 0.9963600028548997|0.9930768681750054
K number (150)| 0.9979301977018057|0.9912925558489758

### 预测

```
$ python BinaryClassification/predict.py -h

Usage: predict.py [options]

Options:
  -h, --help            show this help message and exit
  -f FILE, --file=FILE  data file
  -c CLASSIFIER, --classifier=CLASSIFIER
                        classifier type：'lg' or 'svm'
  -n NGRAM, --ngram=NGRAM
                        the number of n gram
  -u USE, --use=USE     weather use keams
  -k KMEANS, --kmeans=KMEANS
                        the number of kmeanss
```

## LSTM Sequence Classification
## 模型方法
目前很多研究表明，深度学习和神经网络在图像识别和自然语言处理（NLP）方面表现出众。可以可以利用神经网络的LSTM来处理这个分类问题。本方法是利用了基于Google的Tensorflow底层框架上的keras来做的模型。

使用LSTM RNN二元分类法则意味着是在模型上应用监督学习算法。因此，训练数据集中的每个日志条目都需要有一个附带的标签来描述该记录的请求是正常的还是尝试注入攻击的。

## 流程
1. 文本进行预处理。将请求日志文本中的每个字符映射到字典的对应数字中。

2. 采用LSTM进行数据传递。

3. 最后通过全连接输出结果。

## 训练
```
$ cd LSTM/
$ python train.py -h
Usage: train.py [options]

Options:
  -h, --help            show this help message and exit
  -f FILE, --file=FILE  data file
```
## 验证
```
$ python vali_dev.py -h
Usage: vali_dev.py [options]

Options:
  -h, --help            show this help message and exit
  -f FILE, --file=FILE  data file
```

```
$ python vali_waf.py -h
Usage: vali_waf.py [options]

Options:
  -h, --help            show this help message and exit
  -g GOOD, --good=GOOD  good queries file
  -b BAD, --bad=BAD     bad queries file
```

```
$ python vali_signal_sample.py <you tested url>
```
