import numpy
import json
import optparse
from collections import OrderedDict

import pandas
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def predict(csv_file):
    numpy.set_printoptions(suppress=True)

    # 载入字典
    with open('./build/word-dictionary.json') as dict_f:
        dict = ''
        line = dict_f.read()
        dict_json = json.loads(line, object_pairs_hook=OrderedDict)
        for item in dict_json:
            dict = dict + item
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(dict)

    # 载入数据
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.values

    # 预处理
    test_x = dataset[:, 0]
    test_y = dataset[:, 1]
    for index, item in enumerate(test_x):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        test_x[index] = json.dumps(reqJson, separators=(',', ':'))

    test_x = tokenizer.texts_to_sequences(test_x)
    test_process = sequence.pad_sequences(test_x, maxlen=1024)

    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 评估结果
    score, acc = model.evaluate(test_process, test_y, verbose=1, batch_size=128)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))
    print('score: ' + str(score))

    prediction = model.predict(test_process)
    with open("vali_dev_output.csv", 'w') as output:
        output.write('predict,real,url\n')
        for i in range(len(prediction)):
            output.write(str('%.2f' % prediction[i]) + ',' + str(dataset[i, 1]) + ',' + dataset[i, 0] + '\n')


if __name__ == '__main__':
    '''
        为json数据版本的文件检测脚本
        sample : python vali_signal_sample.py test.csv
        return : 0～1之间的数值
    '''
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'data/dev-access.csv'

    predict(csv_file)
