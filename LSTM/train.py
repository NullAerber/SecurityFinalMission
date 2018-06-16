import sys
import os
import json
import pandas
import numpy as np
import optparse
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict


def train(csv_file):
    # 加载数据集
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    # 打乱样本
    dataset = dataframe.sample(frac=1).values

    # X是数据集第一列，请求url纯文本
    # Y是数据第二列，label，{0,1}
    X = dataset[:, 0]
    Y = dataset[:, 1]

    # 清洗文本数据
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for index, item in enumerate(X):
        # 解析json元素
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        # del用于list列表操作，删除一个或者连续几个元素
        # 删除request请求中的无用信息
        # 只保留method, query, path, statuscode, requestPayload{username, password}
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        # 替换原有X[index]
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    # tokenizer是一个分词器，获取当前数据集中所有的字符集合，并构建一个字典
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)

    # 保存字典，若此字典不存在则创建
    word_dict_file = 'build/word-dictionary.json'
    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))
    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

    dict_num = len(tokenizer.word_index) + 1
    # 固定长度，最长1024
    max_log_length = 1024

    # 将X文本数据转换成数字index
    X = tokenizer.texts_to_sequences(X)
    X_processed = sequence.pad_sequences(X, maxlen=max_log_length)

    # 训练集前75%，测试集后25%
    train_size = int(len(dataset) * .75)
    X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    # 调用TensorBoard进行可视化，进行log记录
    tb_callback = TensorBoard(log_dir='./logs', embeddings_freq=1)

    # Sequential序列号模型
    # Sequential的第一层需要接受一个关于输入数据shape的参数
    # 嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
    # Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
    # 模型结构：
    #   嵌入层
    #   Dropout
    #   LSTM
    #   Dropout
    #   Dense全连接层
    model = Sequential()
    model.add(Embedding(input_dim=dict_num, output_dim=32, input_length=max_log_length))
    model.add(Dropout(0.5))
    model.add(LSTM(units=64, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    # 逻辑回归，采用对数损失函数
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=128, epochs=3, validation_split=0.25, callbacks=[tb_callback])

    # Evaluate model
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=128)

    print("Model Accuracy: {:0.2f}%".format(acc * 100))

    # Save model
    model.save_weights('securitai-lstm-weights.h5')
    model.save('securitai-lstm-model.h5')
    with open('securitai-lstm-model.json', 'w') as outfile:
        outfile.write(model.to_json())


if __name__ == '__main__':
    # 命令行参数
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    # 若命令中有指定训练集的目标文件夹
    if options.file is not None:
        csv_file = options.file
    # 若没有,则采用默认原始数据
    else:
        csv_file = 'data/dev-access.csv'
    # 进行训练
    train(csv_file)
