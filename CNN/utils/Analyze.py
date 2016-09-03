import numpy as np
from ..core_layers.ActivationEstimationLayer import MergedDenseActivationLayer, DenseActivationLayer

def get_dense_activation_parameters(model):
    weight = []
    bias = []
    for index, layer in enumerate(model.layers):
        if isinstance(layer, MergedDenseActivationLayer) or isinstance(layer, DenseActivationLayer):
            w, b = model.get_layer_params(layer=index)
            weight.append(w.reshape((1,-1)))
            bias.append(b.reshape((1,-1)))
    return weight, bias

