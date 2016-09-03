import cPickle
import gzip

from keras.datasets import cifar100

import MakeData


def loadMnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    trainingData    = MakeData.shared(train_set)
    validationData  = MakeData.shared(valid_set)
    testData        = MakeData.shared(test_set)
    return trainingData, validationData, testData

def loadCifar100():
    train_set, test_set = cifar100.load_data()
    X, Y = train_set
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    Y = Y.reshape((Y.shape[0],))

    trainingData = MakeData.shared((X[0:40000, :], Y[0:40000]))
    validationData = MakeData.shared((X[40000:50000, :], Y[40000:50000]))

    X, Y = test_set
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    Y = Y.reshape((Y.shape[0],))
    testData = MakeData.shared((X, Y))

    return trainingData, validationData, testData