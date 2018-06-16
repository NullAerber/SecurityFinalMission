import json
import optparse
from collections import OrderedDict
import numpy
import pandas
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def predict(goodqueries, badqueries):
    print("use file:" + good_file + "   " + bad_file)
    # 载入字典
    with open('./build/word-dictionary.json') as dict_f:
        dict = ''
        line = dict_f.read()
        dict_json = json.loads(line, object_pairs_hook=OrderedDict)
        for item in dict_json:
            dict = dict + item
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(dict)

    # 加载waf项目的数据
    df_white = pandas.read_csv(goodqueries, engine='python', sep='!@#$%^&*', header=0).sample(n=50000)
    df_black = pandas.read_csv(badqueries, engine='python', sep='!@#$%^&*', header=0)
    df_white['label'] = 0
    df_black['label'] = 1
    dataset = df_black.append(df_white)
    dataset = dataset.sample(n=10000)
    X_waf = dataset['url'].values.astype('str')
    Y_waf = dataset['label'].values.astype('str')

    # 处理数据
    X_sequences = tokenizer.texts_to_sequences(X_waf)
    X_processed = sequence.pad_sequences(X_sequences, maxlen=1024)

    # 加载模型
    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 评估结果
    score, acc = model.evaluate(X_processed, Y_waf, verbose=1, batch_size=128)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))
    print('score: ' + str(score))

    prediction = model.predict(X_processed)

    with open("vali_waf_output.csv",'w') as output:
        output.write('predict,real,url\n')
        for i in range(len(prediction)):
            output.write(str('%.2f' % prediction[i])+ ',' + Y_waf[i] + ',' + X_waf[i] + '\n')


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-g', '--good', action="store", dest="good", help="good queries file")
    parser.add_option('-b', '--bad', action="store", dest="bad", help="bad queries file")
    options, args = parser.parse_args()

    good_file = 'data/goodqueries.txt'
    bad_file = 'data/badqueries.txt'

    if options.good is not None:
        good_file = options.good
    if options.bad is not None:
        bad_file = options.bad

    predict(good_file, bad_file)
