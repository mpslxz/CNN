import numpy as np

import theano
import theano.tensor as T

import CNN.core_layers.DropoutLayer
import CNN.utils.activations


#   Defining the fully-connected layer class.
#   Class constructor:
#   FullyConnectedLayer(numberOfInputs, numberOfOutputs, activationFnc)
#           activationFnc   = ReLU, sigmoid, tanh, linear

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=CNN.utils.activations.sigmoid, p_dropout=0.0):
        self.trainable = True
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        # initialize weights and biases from a normal distribution
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    # Setting input to the FullyConnectedLayer
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))


        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

        # There is NO dropout in the output
        self.output_dropout = self.output

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))
