import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

class ReLU:
    def __init__(self):
        self.zero_index = None

    def forward(self, x):
        out = x.copy()
        self.zero_index = (x <= 0)
        out[self.zero_index] = 0
        return x

    def backward(self, dout):
        dout[self.zero_index] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class Affine:
    def __init__(self, W, b):
        #   only Linear Layers have params
        self.dW = None
        self.db = None
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        return x@self.W + self.b

    def backward(self, dout):
        dx = dout@self.W.T
        self.dW = self.x.T@dout
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx