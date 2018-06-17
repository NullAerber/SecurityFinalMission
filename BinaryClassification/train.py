import optparse
import pickle
import time
import urllib
import sys

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# 打印时间
def print_time(word):
    a = time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime(time.time()))
    print(a + str(word))


# 训练模型基类
class BaseModel(object):
    def __init__(self,
                 classifier,
                 good,
                 bad,
                 k,
                 n_gram,
                 use_keams):
        self.good = good
        self.bad = bad
        self.k = k
        self.n_gram = n_gram
        self.use_keams = use_keams
        self.classifier = classifier

    def get_name(self):
        return 'base model'

    def train(self):
        print_time('读入数据文件，good：' + self.good + ' bad:' + self.bad)
        x_data = self.load_data()
        print_time('读取结束, good numbers:' + str(len(x_data[0])) + ' bad numbers:' + str(
            len(x_data[1])) + ' total numbers:' + str(len(x_data[0] + x_data[1])))

        # 标记
        good_y = [0 for i in range(len(x_data[0]))]
        bad_y = [1 for i in range(len(x_data[1]))]
        y = good_y + bad_y

        # 向量化
        # 定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 把不规律的文本字符串列表转换成规律的 ( [i,j],weight) 的矩阵X
        # 矩阵元素[(i,j) weight] 表示编号 为 j 的词片 在编号为 i 的url下的 fd-idf 值（weight）
        X = self.vectorizer.fit_transform(x_data[0] + x_data[1])
        print_time('向量化后的维度：' + str(X.shape))
        # 通过kmeans降维 返回降维后的矩阵
        if self.use_keams:
            X = self.transform(self.kmeans(X))
            print_time('降维完成')

        # 使用 train_test_split 分割 X y 列表 testsize表示测试占的比例 random为种子
        # X_train,Y_train  -->> 用来训练模型
        # X_test,Y_test    -->> 用来测试模型的准确性
        print_time('划分测试集训练集')
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print_time('划分完成，开始训练')
        print_time(self.classifier)
        self.classifier.fit(X_train, Y_train)

        # 使用测试值 对 模型的准确度进行计算
        print_time(self.get_name() + '在测试集上模型的准确度:{}'.format(self.classifier.score(X_test, Y_test)))

        # 保存训练结果
        with open('model/' + self.get_name() + '.pickle', 'wb') as output:
            pickle.dump(self, output)

    def predict(self, X):
        # 加载模型
        try:
            with open('model/' + self.get_name() + '.pickle', 'rb') as input:
                self = pickle.load(input)
            print_time('加载 ' + self.get_name() + '模型成功')
        except FileNotFoundError:
            print_time('不存在模型' + self.get_name())
            print_time('请先训练')
            sys.exit(1)

        print_time('数据预处理')

        # 预处理数据
        X = [urllib.parse.unquote(url) for url in X]
        X_processed = self.vectorizer.transform(X)
        if self.use_keams:
            print_time('采用数据降维')
            X_processed = self.transform(X_processed.tolil().transpose())

        print_time('预处理完成,开始预测')
        Y_prediction = self.classifier.predict(X_processed)
        print_time('预测完成,总数：' + str(len(Y_prediction)))
        print_time('good query的数量: ' + str(len(Y_prediction[Y_prediction == 0])))
        print_time('bad query的数量: ' + str(len(Y_prediction[Y_prediction == 1])))
        print_time('good 占比为: ' + str('%0.4f' % (len(Y_prediction[Y_prediction == 0]) / len(Y_prediction))))
        print_time('bad 占比为: ' + str('%0.4f' % (len(Y_prediction[Y_prediction == 1]) / len(Y_prediction))))
        print_time("预测的结果列表:输出至 predict.csv 中")
        with open("predict.csv", 'w') as output:
            output.write('prediction,url\n')
            for i in range(len(Y_prediction)):
                output.write(str(Y_prediction[i]) + ',' + X[i] + '\n')

    # 通过长度为N的滑动窗口将文本分割为N-Gram序列
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - self.n_gram):
            ngrams.append(tempQuery[i:i + self.n_gram])
        return ngrams

    # kmeans 降维
    def kmeans(self, weight):
        print_time('kmeans之前矩阵大小： ' + str(weight.shape))
        weight = weight.tolil().transpose()

        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('model/' + self.get_name() + '.label', 'r') as input:

                print_time('加载keams分词结果成功')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:0]]

        except FileNotFoundError:

            print_time('重新开始Kmeans聚类')
            clf = KMeans(n_clusters=self.k, precompute_distances=False)
            clf.fit(weight)

            # 保存聚类的结果
            self.label = clf.labels_
            with open('model/' + self.get_name() + '.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        print_time('kmeans 完成,聚成 ' + str(self.k) + '类')

        return weight

    # 转换成聚类后结果 输入转置后的矩阵 返回转置好的矩阵
    def transform(self, weight):
        from scipy.sparse import coo_matrix
        a = set()
        # 用coo存 可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号 label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        newWeight = coo_matrix((data, (row, col)), shape=(self.k, weight.shape[1]))
        return newWeight.transpose()

    # 加载数据集
    def load_data(self):
        with open(self.good, 'r') as f:
            good_query_list = [i.strip('\n') for i in f.readlines()[:]]
        with open(self.bad, 'r') as f:
            bad_query_list = [i.strip('\n') for i in f.readlines()[:]]
        return [good_query_list, bad_query_list]


class LG(BaseModel):
    def get_name(self):
        if self.use_keams:
            return 'LG__n' + str(self.n_gram) + '_k' + str(self.k)
        return 'LG_n' + str(self.n_gram)

    def __init__(self, good, bad, k, n_gram, use_keams):
        # 逻辑回归方法模型
        super().__init__(LogisticRegression(), good, bad, k, n_gram, use_keams)


class SVM(BaseModel):
    def get_name(self):
        if self.use_keams:
            return 'SVM__n' + str(self.n_gram) + '_k' + str(self.k)
        return 'SVM_n' + str(self.n_gram)

    def __init__(self, good, bad, k, n_gram, use_keams):
        # 逻辑回归方法模型
        super().__init__(svm.SVC(), good, bad, k, n_gram, use_keams)


if __name__ == '__main__':
    # 命令行参数
    parser = optparse.OptionParser()
    parser.add_option('-c', '--classifier', action="store", dest="classifier", help="classifier type：'lg' or 'svm'")
    parser.add_option('-g', '--good', action="store", dest="good", help="good queries file")
    parser.add_option('-b', '--bad', action="store", dest="bad", help="bad queries file")
    parser.add_option('-n', '--ngram', action="store", dest="ngram", help="the number of n gram")
    parser.add_option('-u', '--use', action="store", dest="use", help="weather use kmeans")
    parser.add_option('-k', '--kmeans', action="store", dest="kmeans", help="the number of kmeanss")
    options, args = parser.parse_args()

    good = 'data/good1.txt'
    bad = 'data/bad1.txt'
    # bad = 'data/bad_waf.txt'

    n_gram = 2
    use_keams = False
    k = 80

    if options.good is not None:
        good = options.good
    if options.bad is not None:
        bad = options.bad
    if options.ngram is not None:
        n_gram = options.ngram
    if options.use is not None:
        use_keams = True
    if options.kmeans is not None:
        k = int(options.kmeans)

    if options.classifier is not None:
        if options.classifier == 'lg':
            model = LG(good, bad, k, n_gram, use_keams)
        elif options.classifier == 'svm':
            model = SVM(good, bad, k, n_gram, use_keams)
        else:
            print('参数错误！只能选择逻辑回归"lg"和SVM向量机"svm"两种方法')
            sys.exit(1)
    else:
        print('无分类法参数！可以选择逻辑回归"lg"和SVM向量机"svm"两种方法')
        sys.exit(0)

    # 进行训练
    model.train()
