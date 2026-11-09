import numpy as np
from common.initialization import *
from common import Layers
from common.functions import *
from collections import OrderedDict

class SimpleCNN:
    def __init__(
            self, 
            hidden_size: int, 
            output_size: int
    ):
        self.layers = OrderedDict()
        self.layers["Conv1"] = Layers.Convolution(
            kernel_num=30, 
            kernel_height=5, 
            kernel_width=5
        )
        self.layers["ReLU1"] = Layers.ReLU()
        self.layers["Pooling1"] = Layers.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Flatten"] = Layers.Flatten()
        self.layers["Affine1"] = Layers.Affine(layer_size=hidden_size)
        self.layers["ReLU2"] = Layers.ReLU()
        self.layers["Affine2"] = Layers.Affine(layer_size=output_size)
        self.the_last_layer = Layers.SoftmaxWithLoss()
        self.params = dict()
        

    def predict(self, x):
        x_copy = np.copy(x)
        for layer in self.layers.values():
            x_copy = layer.forward(x_copy)
        for layer_name, layer in self.layers.items():
            if layer.learnable==True:
                self.params[layer_name] = layer.learnable_params
        return x_copy

    def loss(self, x, t):
        return self.the_last_layer.forward( self.predict(x), t )

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.the_last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grad = dict()
        for layer_name, layer in self.layers.items():
            if layer.learnable==True:
                grad[layer_name] = layer.grad
        return grad

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        batch_size = t.shape[0]
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y==t) / float(batch_size)
        return acc