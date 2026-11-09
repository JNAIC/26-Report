import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.initialization import *

class ReLU:
    def __init__(self):
        self.zero_index = None
        self.learnable = False

    def forward(self, x):
        out = x.copy()
        self.zero_index = (x <= 0)
        out[self.zero_index] = 0
        return out

    def backward(self, dout):
        dout[self.zero_index] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
        self.learnable = False

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class Convolution:
    def __init__(
            self, 
            kernel_num: int, 
            kernel_height: int, 
            kernel_width: int, 
            is_b_needed: bool=True, 
            stride: int=1, 
            pad: int=0, 
            initialization: function=std_initialization, 
            *args, 
            **kwargs
    ):
        self.learnable = True

        self.hasInited = False
        self.kernel = None
        self.b = None
        self.dkernel = None
        self.db = None

        self.is_b_needed = is_b_needed
        self.kernel_num = kernel_num
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.pad = pad
        self.initialization = initialization
        self.args = args
        self.kwargs = kwargs

        self.col_x = None
        self.col_kernel = None
        self.input_shape = tuple()
        self.N = 0
        self.FN = 0
        self.FH = 0
        self.FW = 0
        self.OH = 0
        self.OW = 0

    def initialize(self, channels: int):
        self.kernel = self.initialization(
            (self.kernel_num, channels, self.kernel_height, self.kernel_width), 
            *self.args, 
            **self.kwargs
        )

        if self.is_b_needed == True:
            self.b = np.zeros(self.kernel_num)
            self.learnable_params = {"kernel":self.kernel, "b":self.b}
        else:
            self.learnable_params = {"kernel":self.kernel}

    def forward(self, x):
        if self.hasInited==False:
            self.initialize(x.shape[1])
            self.hasInited = True
        self.N, C, H, W = x.shape
        self.input_shape = x.shape
        self.FN, C, self.FH, self.FW = self.kernel.shape
        self.OH = int(1 + (H + 2*self.pad - self.FH) / self.stride)
        self.OW = int(1 + (W + 2*self.pad - self.FW) / self.stride)
        self.col_x = im2col(x, self.FH, self.FW, self.stride, self.pad)
        self.col_kernel = self.kernel.reshape(self.FN, -1).T
        out = self.col_x @ self.col_kernel + (self.b if self.is_b_needed==True else 0)
        out = out.reshape(self.N, self.OH, self.OW, -1).transpose(0, 3, 1, 2)
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).reshape(self.N*self.OH*self.OW, self.FN)
        if self.is_b_needed==True:
            self.db = np.sum(dout, axis=0)
        self.dkernel = (self.col_x.T@dout).T.reshape(self.kernel.shape)
        dx = dout@self.col_kernel.T
        dx = col2im(dx, self.input_shape, self.FH, self.FW, self.stride, self.pad)
        return dx

    @property
    def grad(self):
        if self.is_b_needed==True:
            return {'kernel':self.dkernel, 'b':self.db}
        return {'kernel':self.dkernel}
    
class Pooling:
    def __init__(
            self, 
            pool_h, 
            pool_w, 
            stride=1, 
            pad=0
    ):
        self.learnable = False

        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.mask = None
        self.input_shape = tuple()

    def forward(self, x: np.ndarray):
        N, C, H, W = self.input_shape = x.shape
        OH = int(1+(H+2*self.pad-self.pool_h)/self.stride)
        OW = int(1+(W+2*self.pad-self.pool_w)/self.stride)

        col_x = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col_x = col_x.reshape(-1, self.pool_h*self.pool_w)
        self.mask = np.argmax(col_x, axis=1)  # save the max values' positions as well as what we do in the implementation of ReLU's backpropagation
        out = np.max(col_x, axis=1)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray):
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, 1)
        dx = np.zeros(
            (dout.size, self.pool_h*self.pool_w), 
            dtype=dout.dtype
        )
        dx[np.arange(dx.shape[0]), self.mask] = dout.flatten()
        #   input_shape[1] = C
        #   the original shape of col_x(the exact output of im2col) = (-1, C*pool_h*pool_w)
        #   because in CNN, we need to keep the same areas different channels covered by kernel in a single row
        #   but in Pooling, the channels are independent from each other
        #   Therefore, we need to reshape col_x into (-1, pool_h*pool_w) in forward propagation
        #   Here, we need to reshape it back to the original shape of col_x
        dx = dx.reshape(-1, self.input_shape[1]*self.pool_h*self.pool_w)
        dx = col2im(dx, self.input_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class Flatten:
    def __init__(self):
        self.learnable = False
        self.original_shape = tuple()

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)

class Affine:
    def __init__(
            self, 
            layer_size: int, 
            is_b_needed: bool=True, 
            initialization: function=He, 
            *args, 
            **kwargs
    ):
        self.learnable = True
        #   only Linear Layers have params
        self.hasInited = False
        self.dW = None
        self.db = None
        self.W = None
        self.b = None

        self.layer_size = layer_size
        self.is_b_needed = is_b_needed
        self.initialization = initialization
        self.args = args
        self.kwargs = kwargs
        self.learnable_params = dict()
        
        self.x = None

    #   lazy initialization
    def initialize(self, last_layer_size):
        self.W = self.initialization(
            (last_layer_size, self.layer_size), 
            *self.args, 
            **self.kwargs
        )
        if self.is_b_needed:
            self.b = np.zeros(self.layer_size)
            self.learnable_params = {'W':self.W, 'b':self.b}
        else:
            self.learnable_params = {'W':self.W}

    def forward(self, x):
        if self.hasInited==False:
            self.initialize(x.shape[1])
            self.hasInited = True
        self.x = x
        return x@self.W + self.b if self.is_b_needed==True else x@self.W

    def backward(self, dout):
        dx = dout@self.W.T
        self.dW = self.x.T@dout
        if self.is_b_needed==True:
            self.db = np.sum(dout, axis=0)
        return dx

    @property
    def grad(self):
        if self.is_b_needed==True:
            return {'W':self.dW, 'b':self.db}
        return {'W':self.dW}

class BatchNorm:
    def __init__(self, batch_size: int):
        self.learnable = True
        self.batch_size = batch_size
        self.gamma = np.ones(batch_size)
        self.beta = np.zeros(batch_size)
        self.x_normed = None
        self.variance = None
        self.dgamma = None
        self.dbeta = None
        self.learnable_params = {"gamma":self.gamma, "beta":self.beta}

    def forward(self, x):
        mu = np.mean(x, axis=0)
        self.variance = np.var(x, axis=0)
        self.x_normed = (x - mu) / np.sqrt(self.variance + 1e-7)
        return self.gamma*self.x_normed + self.beta

    def backward(self, dout):
        B = dout.shape[0]

        # Gradients of gamma and beta
        self.dgamma = np.sum(dout * self.x_normed, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # Gradient of normalized input
        dx_normed = dout * self.gamma

        # Gradient of input
        dx = (
            B * dx_normed
            - np.sum(dx_normed, axis=0)
            - self.x_normed * np.sum(
                dx_normed * self.x_normed, axis=0
            )
        ) / (B * np.sqrt(self.variance + 1e-7))

        return dx

    @property
    def grad(self):
        return {'gamma':self.dgamma, 'beta':self.dbeta}

class SoftmaxWithLoss:
    def __init__(self):
        self.learnable = False
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