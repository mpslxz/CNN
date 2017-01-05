
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams

# dropout layer for random masking the layer calculations

class DropoutLayer(object):
    def __init__(self, p_dropout=0.1):
        self.p_dropout = p_dropout
        self.trainable = False
        self.params = [None]
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt
        self.output = self.inpt
        self.output_dropout = self._dropout(inpt, self.p_dropout)


    def _dropout(self, inpt, p_dropout):

        # random number generation from a binomial distribution
        srng = shared_randomstreams.RandomStreams(
            np.random.RandomState(0).randint(999999))
        mask = srng.binomial(n=1, p=1-p_dropout, size=inpt.shape)

        # Masking the layer weights
        return inpt*T.cast(mask, theano.config.floatX)