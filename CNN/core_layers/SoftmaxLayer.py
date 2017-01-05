import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import softmax

import CNN.core_layers.DropoutLayer


#   Defining the fully-connected layer class.
#   Class constructor:
#   SoftmaxLayer(numberOfInputs, numberOfOutputs)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.trainable = True
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))

        # Output is masked by 1 - the probability of the dropout layer
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)


        self.output_dropout = self.output

    def cost(self, net):
        # negative log-likelihood cost function
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


