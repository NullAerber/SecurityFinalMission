# SecurityFinalMission

## 环境

> Python 环境： 3.5

## 说明
该项目是课程大作业，只具有实践复现效果，底层理论知识不提供。

是结合了[用机器学习玩转恶意URL检测](http://www.freebuf.com/articles/network/131279.html) 后的改进版本，
加入了[SVM支持向量机](http://www.freebuf.com/articles/web/130004.html)进行对比，
同时加入了以keras为框架的LSTM预测。

本项目相关具体介绍请[下载报告](https://github.com/NullAerber/SecurityFinalMission/paper.pdf)

## 数据集
本项目一共有三大类数据集

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

### 模型结果
.label文件保存的是分词结果，只与数据集有关

.pickle文件是训练后的模型

### 实验结果
训练即均是默认训练集：

good = './BinaryClassification/data/good1.txt'

bad = './BinaryClassification/data/bad1.txt'



# Sequence Classification

0 =正常请求日志样本
1 =请求日志表示尝试的注入攻击

画一个流程图
LSTM神经网络图

paper：Adam: A Method for Stochastic Optimization

https://medium.com/slalom-engineering/detecting-malicious-requests-with-keras-tensorflow-5d5db06b4f28
https://zhuanlan.zhihu.com/p/29768996

http://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
http://keras-cn.readthedocs.io/en/latest/models/sequential/#sequential
http://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/#embedding_1
http://deeplearning.net/tutorial/lstm.html
https://blog.csdn.net/saltriver/article/details/63683092
https://www.zhihu.com/question/27126057
http://www.360doc.com/content/18/0301/04/52389397_733320586.shtml
