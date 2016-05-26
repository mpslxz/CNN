
import numpy as np
import DropoutLayer
import theano
import theano.tensor as T
from FullyConnectedLayer import FullyConnectedLayer
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


class denseActivationLayer(object):

    def __init__(self, n_in, n_hiddens_dense, activation_fcn = sigmoid):
        self.denseEstimators = []
        self.w = theano.shared(np.asarray(size=(), dtype=theano.config.floatX), name='w', borrow=True)
        for neuron in range(0, n_in):
            self.denseEstimators.append(FullyConnectedLayer(n_in=n_hiddens_dense,n_out=1,activation_fn=activation_fcn))


