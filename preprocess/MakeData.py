# Training, validation and test samples are ordered as m x n matrices with m samples of n elements.
# Reshape and stack all of the training images and concatenate their respective labels for the targets.
# Use the shared() function to set the data types of the training data. See the example below:

# trainingSamples   = np.ndarray((numberOfTrainingImages, imageWidth*imageHeight))
# labels            = np.zeros((numberOfTrainingImages))
# for i in range(1, numberOfTrainingImages):
#       trainingSamples[i, :] = image_i
#       labels[i] = classOfImage_i
# trainingData = shared([trainingSamples, labels])

import numpy as np
import theano
import theano.tensor as T

# Setting the type of the data for the network
def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

def sharedForPrediction(data):
        shared_x = theano.shared(
            np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return shared_x