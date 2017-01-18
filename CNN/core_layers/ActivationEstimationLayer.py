import numpy as np

import theano
import theano.tensor as T


def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid


class DenseActivationLayer(object):

    def __init__(self,n_hiddens_dense, activation_fn = sigmoid):
        self.n_hidden_dense = n_hiddens_dense
        self.activation_fn = activation_fn
        self.trainable = True
        self.w_decode = theano.shared(np.array(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_hiddens_dense),
                                                         size=(1, n_hiddens_dense)),
                                        dtype=theano.config.floatX), name='w_decode', borrow=True)
        self.w_encode = theano.shared(np.array(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_hiddens_dense),
                                                         size=(1, n_hiddens_dense)),
                                        dtype=theano.config.floatX), name='w_encode', borrow=True)

        self.b = theano.shared(np.array(np.random.normal(loc=0.0, scale=1.0, size=(n_hiddens_dense,)),
                                        dtype=theano.config.floatX), name='b', borrow=True)

        self.w = theano.shared(np.hstack((self.w_decode.get_value(), self.w_encode.get_value())), name='w')

        self.params = [self.w, self.b]


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        inpt_shape = inpt.shape
        self.inpt = inpt.reshape((-1,1))

        output = T.dot(self.activation_fn(T.dot(self.inpt, self.w_encode) + self.b), T.transpose(self.w_decode))
        self.output = output.reshape(inpt_shape)
        self.output_dropout = self.output


class MergedDenseActivationLayer(object):

    def __init__(self,n_hiddens_dense, activation_fn = sigmoid):
        self.n_hidden_dense = n_hiddens_dense
        self.trainable = True
        self.activation_fn = activation_fn
        self.w = theano.shared(np.array(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_hiddens_dense),
                                                         size=(1, n_hiddens_dense)),
                                        dtype=theano.config.floatX), name='w', borrow=True)
        self.b = theano.shared(np.array(np.random.normal(loc=0.0, scale=1.0, size=(n_hiddens_dense,)),
                                        dtype=theano.config.floatX), name='b', borrow=True)

        self.params = [self.w, self.b]


    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        inpt_shape = inpt.shape
        self.inpt = inpt.reshape((-1,1))

        output = T.dot(self.activation_fn(T.dot(self.inpt, self.w) + self.b), T.transpose(self.w))
        self.output = output.reshape(inpt_shape)
        self.output_dropout = self.output

