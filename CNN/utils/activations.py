import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

def linear(z): return z
def absolute(z): return abs(z)
def ReLU(z): return T.maximum(0.0, z)
def elu(z):
    if T.gt(z,0.0): return z
    else:   return T.exp(z)-1
