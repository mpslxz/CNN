
import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams

# dropout layer for random masking the layer calculations
def dropout_layer(layer, p_dropout):

    # random number generation from a binomial distribution
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)

    # Masking the layer weights
    return layer*T.cast(mask, theano.config.floatX)