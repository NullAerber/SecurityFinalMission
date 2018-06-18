from train import LG
from train import SVM
import optparse
import sys

if __name__ == '__main__':
    # 命令行参数
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    parser.add_option('-c', '--classifier', action="store", dest="classifier", help="classifier type：'lg' or 'svm'")
    parser.add_option('-n', '--ngram', action="store", dest="ngram", help="the number of n gram")
    parser.add_option('-u', '--use', action="store", dest="use", help="weather use keams")
    parser.add_option('-k', '--kmeans', action="store", dest="kmeans", help="the number of kmeanss")
    options, args = parser.parse_args()

    testfile = './BinaryClassification/data/good1.txt'
    n_gram = 2
    use_keams = False
    k = 80

    # 若命令中有指定训练集的目标文件夹
    if options.file is not None:
        testfile = options.file
    if options.ngram is not None:
        n_gram = options.ngram
    if options.use is not None:
        use_keams = True
    if options.kmeans is not None:
        k = int(options.kmeans)

    if options.classifier is not None:
        if options.classifier == 'lg':
            model = LG('', '', k,n_gram, use_keams)
        elif options.classifier == 'svm':
            model = SVM('', '', k,n_gram, use_keams)
        else:
            print('参数错误！只能选择逻辑回归"lg"和SVM向量机"svm"两种方法')
            sys.exit(1)
    else:
        print('无分类法参数！可以选择逻辑回归"lg"和SVM向量机"svm"两种方法')
        sys.exit(0)

    with open(testfile, 'r') as f:
        print('预测数据集： ' + testfile)
        predict_list = [i.strip('\n') for i in f.readlines()[:]]

        model.predict(predict_list)
        print("预测的结果列表:输出至 predict.csv 中")
