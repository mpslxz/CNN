import numpy as np

import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import CNN.utils.activations


#   Defining the convolutional-pooling layer class.
#   Class constructor:
#   ConvPoolLayer(imageShape, filterShape, poolSize, activationFnc)
#           imageShape      = (mini_batch_size, numberOfChannels, width, height)
#           filterShape     = (numberOfFilters, numberOfInputs, filterWidth, filterHeight)
#           pooSize         = (width, height)
#           activationFnc   = ReLU, sigmoid, tanh, linear
#
#   Note: In the convolutional-pooling layers, the output size would be:
#       output_size = (input_size - filter_size + 1) / pool_size


class ConvPoolLayer(object):

    def __init__(self, filter_shape, image_shape, conv_type='valid', poolsize=(2, 2),
                 activation_fn=CNN.utils.activations.sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        self.border_mode = conv_type
        # initialize weights and biases from a normal distribution

        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    # Setting input to the ConvPoolLayer
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)

        # Convolving the input with the filters
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape, border_mode=self.border_mode)

        # Down sampling with respective pooling size
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)

        # Calling the activation function with down sampled feature map as the input

        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # no dropout in the convolutional layers
        self.output_dropout = self.output
