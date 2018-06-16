import json
import optparse
from collections import OrderedDict

from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def predict(signal_sample):
    # 载入字典
    with open('./build/word-dictionary.json') as dict_f:
        dict = ''
        line = dict_f.read()
        dict_json = json.loads(line, object_pairs_hook=OrderedDict)
        for item in dict_json:
            dict = dict + item
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(dict)

    # 预处理数据
    test_process = tokenizer.texts_to_sequences([signal_sample])
    test_process = sequence.pad_sequences(test_process, maxlen=1024)

    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict(test_process)
    print(signal_sample + ":" + str(prediction))

if __name__ == '__main__':
    '''
        为单样本检测脚本，请求url直接在命令参数中
        sample : python vali_signal_sample.py 'login/username/'
        return : 0～1之间的数值
    '''
    parser = optparse.OptionParser()
    options, args = parser.parse_args()

    if args[0] is not None:
        predict(args[0])
