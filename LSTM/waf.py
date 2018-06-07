import sys
import os
import json
import pandas
import numpy
import optparse
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

def train():
    # 加载另一组的数据
    df_black = pandas.read_csv('data/badqueries.txt',engine='python',sep='!@#$%^&*',header=0)
    df_white = pandas.read_csv('data/goodqueries.txt',engine='python',sep='!@#$%^&*',header=0).sample(n=50000)
    df_black['label'] = 1
    df_white['label'] = 0
    new_dataset = df_black.append(df_white)
    new_dataset = new_dataset.sample(n=10000)
    X_waf = new_dataset['url'].values.astype('str')
    Y_waf = new_dataset['label'].values.astype('str')

    dataframe = pandas.read_csv('data/dev-access.csv', engine='python', quotechar='|', header=None)
    dataset = dataframe.values

    # Preprocess dataset
    X = dataset[:,0]
    for index, item in enumerate(X):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        X[index] = json.dumps(reqJson, separators=(',', ':'))
        
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)
    
    max_log_length = 1024

    X_sequences = tokenizer.texts_to_sequences(X_waf)
    X_processed = sequence.pad_sequences(X_sequences, maxlen=max_log_length)

    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Evaluate model
    score, acc = model.evaluate(X_processed, Y_waf, verbose=1, batch_size=128)
    print("Model Accuracy: {:0.2f}%".format(acc * 100))
    print(score)
    prediction = model.predict(X_processed)
    print (prediction[0])


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    options, args = parser.parse_args()

    train()
