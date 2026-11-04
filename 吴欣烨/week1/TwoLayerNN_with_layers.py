import numpy as np
from common.Layers import *
from collections import OrderedDict

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)*weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)*weight_init_std
        self.params['b2'] = np.zeros(output_size)
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        z = self.predict(x)
        return self.last_layer.forward(z, t)

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        gradient = {}
        gradient['dW1'] = numerical_gradient(loss_w, self.params['W1'])
        gradient['db1'] = numerical_gradient(loss_w, self.params['b1'])
        gradient['dW2'] = numerical_gradient(loss_w, self.params['W2'])
        gradient['db2'] = numerical_gradient(loss_w, self.params['b2'])
        return gradient
    
    def gradient(self, x, t):
        #   forward first(init the status of each layers)
        self.loss(x, t)
        grad = {}
        dout = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad['dW1'] = self.layers['Affine1'].dW
        grad['db1'] = self.layers['Affine1'].db
        grad['dW2'] = self.layers['Affine2'].dW
        grad['db2'] = self.layers['Affine2'].db
        return grad

    def optimize(self, x, t, lr=0.01):
        grad = self.gradient(x, t)
        for (key, value) in self.params:
            value -= lr*grad[f"d{key}"]

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        batch_size = t.shape[0]
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y==t) / float(batch_size)
        return acc

