import numpy as np
from common.functions import *
from common import Layers
from common.initialization import *
from collections import OrderedDict

class MultiLayersNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int, initialization: function=He):
        self.layers = OrderedDict()
        self.layers["Affine1"] = Layers.Affine(input_size, hidden_size, initialization)
        self.layers["BN1"] = Layers.BatchNorm(batch_size)
        self.layers["ReLU1"] = Layers.ReLU()
        self.layers["Affine2"] = Layers.Affine(hidden_size, hidden_size, initialization)
        self.layers["BN2"] = Layers.BatchNorm(batch_size)
        self.layers["ReLU2"] = Layers.ReLU()
        self.layers["Affine3"] = Layers.Affine(hidden_size, hidden_size, initialization)
        self.layers["BN3"] = Layers.BatchNorm(batch_size)
        self.layers["ReLU3"] = Layers.ReLU()
        self.layers["Affine4"] = Layers.Affine(hidden_size, output_size, initialization)
        self.last_layer = Layers.SoftmaxWithLoss()
        self.params = dict()
        for layer_name, layer in self.layers.items():
            if layer.learnable:
                self.params[layer_name] = layer.learnable_params

    def forward(self, x):
        x_copy = np.copy(x)
        for layer in self.layers.values():
            x_copy = layer.forward(x_copy)

        return x_copy

    def loss(self, x, t):
        pred = self.forward(x)
        return self.last_layer.forward(pred, t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = dict()
        for layer_name, layer in self.layers.items():
            if layer.learnable:
                grad[layer_name] = layer.grad
        return grad

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        batch_size = t.shape[0]
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y==t) / float(batch_size)
        return acc