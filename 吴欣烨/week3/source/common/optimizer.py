import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params: dict, gradient: dict):
        for layer_name, layer_params in params.items():
            for param_name, param_val in layer_params.items():
                param_val -= self.lr * gradient[layer_name][param_name]


class Momentum:
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.v = dict()

    def step(self, params, gradient):
        if self.v == {}:
            for layer_name, layer_params in params.items():
                self.v[layer_name] = dict()
                for param_name, param_val in layer_params.items():
                    self.v[layer_name][param_name] = np.zeros_like(param_val)

        for layer_name, layer_params in params.items():
            for param_name, param_val in layer_params.items():
                self.v[layer_name][param_name] = self.alpha * self.v[layer_name][param_name] - self.lr * gradient[layer_name][param_name]
                param_val += self.v[layer_name][param_name]
        # for key in params.keys():
        #     self.v[key] = self.alpha * self.v[key] - self.lr * gradient[f"d{key}"]
        #     params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = dict()

    def step(self, params, gradient):
        if self.h == {}:
            for layer_name, layer_params in params.items():
                self.h[layer_name] = dict()
                for param_name, param_val in layer_params.items():
                    self.h[layer_name][param_name] = np.zeros_like(param_val)
            # for key, val in params.items():
            #     self.h[key] = np.zeros_like(val)

        for layer_name, layer_params in params.items():
            for param_name, param_val in layer_params.items():
                self.h[layer_name][param_name] += gradient[layer_name][param_name]**2
                param_val -= self.lr * gradient[layer_name][param_name] / np.sqrt(self.h[layer_name][param_name] + 1e-7)
        # for key in params.keys():
        #     self.h[key] += gradient[f"d{key}"]**2
        #     params[key] -= self.lr * gradient[f"d{key}"] / np.sqrt(self.h[key] + 1e-7)

class RMSprop:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.h = dict()

    def step(self, params, gradient):
        if self.h == {}:
            for layer_name, layer_params in params.items():
                self.h[layer_name] = dict()
                for param_name, param_val in layer_params.items():
                    self.h[layer_name][param_name] = np.zeros_like(param_val)
        # if self.h == {}:
        #     for key, val in params.items():
        #         self.h[key] = np.zeros_like(val)
        for layer_name, layer_params in params.items():
            for param_name, param_val in layer_params.items():
                self.h[layer_name][param_name] = self.beta*self.h[layer_name][param_name] + (1-self.beta)*(gradient[layer_name][param_name]**2)
                param_val -= self.lr * gradient[layer_name][param_name] / np.sqrt(self.h[layer_name][param_name] + 1e-7)

        # for key in params.keys():
        #     self.h[key] = self.beta*self.h[key] + (1-self.beta)*(gradient[f"d{key}"]**2)
        #     params[key] -= self.lr * gradient[f"d{key}"] / np.sqrt(self.h[key] + 1e-7)

class Adam:
    def __init__(self, lr=0.01, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = dict()
        self.v = dict()
        self.t = 0

    def step(self, params, gradient):
        if self.m == {} or self.v == {}:
            for layer_name, layer_params in params.items():
                self.m[layer_name] = dict()
                self.v[layer_name] = dict()
                for param_name, param_val in layer_params.items():
                    self.m[layer_name][param_name] = np.zeros_like(param_val)
                    self.v[layer_name][param_name] = np.zeros_like(param_val)

        # if self.m == {} or self.v == {}:
        #     for key, val in params.items():
        #         self.m[key] = np.zeros_like(val)
        #         self.v[key] = np.zeros_like(val)

        self.t += 1
        for layer_name, layer_params in params.items():
            for param_name, param_val in layer_params.items():
                self.m[layer_name][param_name] = self.beta_1*self.m[layer_name][param_name] + (1-self.beta_1)*gradient[layer_name][param_name]
                self.v[layer_name][param_name] = self.beta_2*self.v[layer_name][param_name] + (1-self.beta_2)*(gradient[layer_name][param_name]**2)
                m_corrected = self.m[layer_name][param_name] / (1-self.beta_1**self.t)
                v_corrected = self.v[layer_name][param_name] / (1-self.beta_2**self.t)
                param_val -= self.lr*m_corrected / np.sqrt(v_corrected + 1e-7)

        # self.t += 1
        # for key in params.keys():
        #     self.m[key] = self.beta_1*self.m[key] + (1-self.beta_1)*gradient[f"d{key}"]
        #     self.v[key] = self.beta_2*self.v[key] + (1-self.beta_2)*(gradient[f"d{key}"]**2)
        #     m_corrected = self.m[key] / (1-self.beta_1**self.t)
        #     v_corrected = self.v[key] / (1-self.beta_2**self.t)
        #     params[key] -= self.lr*m_corrected / np.sqrt(v_corrected + 1e-7)