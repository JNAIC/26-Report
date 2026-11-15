#前言：我们已经使用了torch来构造人工神经网络，但是里面的所有工具都是他人写好的，
# 为了跟加透彻地了解底层工作原理，我们将要使用numpy这一基础得多的库来尝试构建一个人工神经网络
#目标：接下来要构建的人工神经网络与hello_world_in_NN_learning.py中的内容相同，这里不再赘述
#下面的代码分为四个部分：
#1.functions：常用的一些函数
#2.Layers:线性网络常用的激活函数
#3.真正的线性神经网络

#functions
import numpy as np
#交叉熵损失计算函数
#首先我们来了解一下信息论的经典概念，信息熵，然后，我们才能理解什么叫交叉熵
#从而理解这里交叉熵损失计算对人工神经网络训练的意义
#信息熵的概念来源于一个非常实际的问题：如何对信息进行有效压缩从而增加信息传递的效率？
#我们首先设想这样一个场景：
#在冥王星上有一个探测器，我们能对它发出四种不同的指令，up,down,right,left。这些指令必须使用0和1来编码
#由于长距离通信不便，我们需要尽可能地压缩发出的信息量。那么我们应该如何编码这四种指令
#已知因为探测计划的需求，我们的四种指令不是乱发的：而是有一定的规律。在发出的指令中，up占1/2，down占1/4，right占1/8，left占1/8.
#首先我们来看最直接的思路：给每个指令分配两个bit；
#up,down,right,left
#00, 01 , 10  , 11
#这样的话每条指令的平均bit就会是2*4/4=2
#而另一中自然而然的想法是给常用的指令分配较少的bit，给不常用的指令分配较多的bit
#我们就会有一种新的编码方式：
#up,down,right,left
#1, 01 , 000  , 001
#这样的话每条指令的平均bit就会是：1*0.5 +2*0.25 +3*2*0.125 = 1.75
#确实提高了效率！
#（注意在信息传递的问题中，我们发出的信息同时要能提供解码方式，第一种编码方式的解码方式显而易见，但第二种编码方式实际上也有确定的解码方式，找一找）
#在对这种问题的分析中，香农提出了信息熵的概念：
#在信息压缩问题中，为发生概率高的事件分配较少的信息量，而为发生概率较低的事件分配较多的信息量是压缩信息的一种合理做法
#而在最常用的二进制中，一个事件在最优压缩效率下应该分配的信息量的期望值就是它的信息熵
#计算方法为-log2(P(X)),P(X)代表概率
#那么信息熵与人工智能之间有什么关系呢？
#一种思路是：对信息进行压缩的过程就是要尽量利用信息内存在的规律（所以噪声是无法压缩的，同时完美压缩的信息看起来也应该像噪声一样）
#而寻找规律恰好是智能的最重要特征之一！
#从这个思路出发我们可以得到一个著名论断：压缩即智能！
#接下来我们介绍交叉熵
#所谓交叉熵，就是一种信息压缩方式在另外一种信息分布下的表现：
#还是那个探测器的例子，但这次，因为探测计划的改变，我们发送的指令的分布发生了变化up占1/8，down占1/8，right占1/4，left占1/2
#在这种情况下，显然信息压缩不再高效,假设新分布是Y,Yn代表分布中的某一具体的项
#交叉熵就可以用sum(P(Y)*(-log2(P(X))))来计算
#如果根据我们前面提到的“对信息进行压缩的过程就是要尽量利用信息内存在的规律”的思路来看，交叉熵就是在衡量一种规律在另一种情景下的适用程度
#而在深度学习的任务中，我们希望学得的模型的输出的规律与训练语料的内在规律相同
#这时使用交叉熵来作为评估训练结果的损失函数就变成了一种自然而然的选择
def cross_entropy_error(y, t):
    #python内部有一个内置对象memoryview,它提供了一种无需拷贝就能访问另一对象内部原始内存的方式。
    #bytes，byte arrayNumpy数组都可以用它来访问
    #但这里不是，这里直接访问的是ndarray的属性ndim就是维度数，也就是shape的长度，当ndim为0时，它代表标量
    if y.ndim == 1:#如果是一维的数组，就将其转化为二维Numpy数组，加了一个维度，方便下一步的argmax()，
        # pytorch中使用unsqueeze()来插入维度
        #这里numpy的API也可以使用np.newaxis()
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)#argmax()函数，求最大值的下标索引值，这里沿着序号为1的轴，实际上是第二个轴来求索引

    batch_size = y.shape[0]
    #下面这一行比较复杂，我们拆开来看
    #首先经过上面的过程后，t是一个一维数组
    #np.arange()也会创建一个一维数组
    #这里是Numpy中一种经典的高级索引用法：按行取对应列，为每个样本取出它自己那一列的元素
    #这里另外举一个例子：
    # y = np.array([[0.1, 0.7, 0.2],   # 样本0 的各类分数
    #               [0.8, 0.1, 0.1],   # 样本1
    #               [0.2, 0.3, 0.5]])  # 样本2
    # t = np.array([1, 0, 2])          # 每个样本的真实类别
    # batch_size = 3
    #
    # result = y[np.arange(batch_size), t]
    # 结果: array([0.7, 0.8, 0.5])
    #所以最里面会得到
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#对于交叉熵，这里再补充几点：
#1.ln求导计算简单同时它与log2只相差一个常数倍，这里的差异可以被吸收到学习率当中，这样一来上面的函数就直接使用了ln
#2.在香农的信息熵的公式中，使用log2来作为计算函数是自然而然的，它来自于编码空间随信息量而指数增长的特点
#但是在深度学习领域，这里可没有什么编码空间
#那为什么对数函数就一定是最好的函数呢？在(0,1)范围内递减的函数明明有无数多个
#在选择损失函数时，我们希望它能正确地反映模型在整体地训练数据上的训练效果
#注意模型并不只输出一个结果，而是输出所有可能的结果的概率分布，
#我们要习得的“规律”也不只是下一个token，而是整个训练集上的分布规律
#在局部、严格恰当评分规则的前提下，数学上可以证明：如果想要当输出的分布等于训练集中的token分布是损失函数取得最小值，也就是训练成果最优，那就必须选择对数函数
#但具体的证明过程超出了这里的讨论范围，略过不提
#3.交叉熵计算在模型蒸馏中能起到至关重要的作用
#因为蒸馏过程中我们的“答案”不再是单个的确定结果，而是被蒸馏模型给出的完整分布
#就好比请了一位高手来讲解所有的好棋一样
#我们能以惊人的效率训练出表现相近的模型
#尽管新训练的模型的参数量可能要小得多

#线性激活函数
#对输入不做任何处理，直接输出
#这里我们仍然将其封装为函数，尽管这看起来好像没有任何意义
#因为这可以在后续使代码更加规范，同时增强可读性
def linear(a):
    return a

#sigmoid函数，又名logistic曲线
#能将输入压缩到(0,1)的区间内
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

#归一化函数
#输出值的总和总是1
#当模型的输出是概率分布时非常有用
def softmax(a):
    if a.ndim == 2:                          # batch：按行归一化
        a = a - np.max(a, axis=1, keepdims=True)
        e = np.exp(a)
        return e / np.sum(e, axis=1, keepdims=True)
    a = a - np.max(a)
    e = np.exp(a)
    return e / np.sum(e)

#数值梯度函数，用于计算多元函数在点x处的数值梯度
def numerical_gradient(f, x):
    #微小步长h
    h = 1e-4
    #初始化一个用来存储算出的梯度的numpy数组，初始值为0，形状与输入x相同
    grad = np.zeros_like(x)
    #使用nditer遍历数组
    #它是Numpy提供的一个高效多维数组迭代器对象，它让我们能够使用统一，灵活的方式遍历每一个数组元素
    #这里传入了三个参数。一个是被迭代对象，一个是告诉迭代器要追踪多维索引，最后声明允许对正在遍历的数组X进行读写操作
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:#nditer的属性finished，数组是否被遍历完全
        #使用当前元素的多维索引保存原值
        idx = it.multi_index
        tmp_val = x[idx]
        #对变量做中心差分，从而计算梯度
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        #切换到下一个元素
        it.iternext()
    #返回算好的梯度
    return grad

#Layers
#线性层
import sys, os
#将父目录加入python的模块搜索路径，以便能够import位于上一级目录中的模块
#从而实现跨脚本程序
#deep-learning-from-scratch/      ← 项目根目录
# ├── dataset/
# │   └── mnist.py
# ├── common/
# │   └── functions.py
# └── ch04/
#     └── gradient.py
sys.path.append(os.pardir)
#relu激活函数，负数输出为零，正数输出为原值
#这里的每一个类中都定义了前向过程与反向过程
class ReLU:
    def __init__(self):
        #定义一个属性，用来存储将要被修改为0的元素
        self.zero_index = None
    #前向过程就是修改了x，并返回
    def forward(self, x):
        out = x.copy()
        self.zero_index = (x <= 0)
        out[self.zero_index] = 0
        return out
    #反向过程
    #反向过程在反向传播算法中使用用来计算梯度
    #这里传入的dout就是上游计算出来的梯度
    #因为relu只是将负数置为0
    #所以对正数，梯度不变
    #对负数，梯度置为0
    def backward(self, dout):
        dout[self.zero_index] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    #正向过程用来将差距很大的数据压缩到(0,1)区间内
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    #反向过程就是应用链式法则，传入的梯度就是对内函数的导数
    #我们只要再乘上对外函数求导的结果即可
    #这里的表达式并不是那么显然
    #但仔细证明一遍就会发现(1.0-self.out)*self.out确实是对sigmoid求导的结果
    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx
#仿射层
#仿射变换的英文就是Affine Transformation
#在深度学习领域，它又叫全连接层或者线性层
class Affine:
    #初始化过程接收两个参数数组，权重矩阵和偏置
    def __init__(self, W, b):
        self.dW = None
        self.db = None
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        #矩阵乘法，同时加上bias
        return x@self.W + self.b

    def backward(self, dout):
        #这里.T是行列转置的意思
        #在这里，至少我这种初学者在对矩阵乘法尚且都用不熟悉的情况下，
        #如果不好好写明白前向和反向过程中矩阵发生的变化的话，我一定会被绕晕
        #首先我们来看前向过程：
        #假设X的shape是(N,D)，N个样本，每个样本有D个特征
        #那么W的shape就应该是(D,H)输入维度为D，输出维度为H
        #相乘之后输出值的shape就是(N,H)
        #如果放到一个确定的情景中来：假设我们在使用这个类来构建一个多层感知机（MLP）
        #那么N就是batch_size，D就是前一层的神经元数目，H就是后一层的神经元数目
        #到了反向传播过程：
        #我们要将梯度传向前一层，这时就需要行列转置
        #dout的shape是(N,H)
        #要传向上一层，我们需要(N,H)@(H,D)，才能得到结果(N,D)
        dx = dout@self.W.T
        #同理计算权重的梯度时：self.x是(N,D),dout是(N,H)
        #计算(D,N)@(N,H)就会得到(D,H),与W的shape完美契合，就可以用来更新函数了
        self.dW = self.x.T@dout
        self.db = np.sum(dout, axis=0)
        #最后返回前一层的梯度，完成这一层的反向传播与梯度计算
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
        if self.t.size == self.t.ndim:  # 标签形式
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        else:  # one-hot
            dx = (self.y - self.t) / batch_size
        return dx
#TwoLayerNN
#这一段在函数等各种工具都构建好了的情况下，和使用pytorch等工具库构建神经网络已经没有太大区别
#这里就不再详细注释，简单看看就好
#主要要注意的是这里的代码的实现核心是字典，与pytorch的面向对象实现有一些不同
from collections import OrderedDict

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
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
        for (key, value) in self.params.items():
            value -= lr * grad[f"d{key}"]

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        batch_size = t.shape[0]
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(batch_size)
        return acc


#optimizers
#注意下面的代码都只是这些优化器的其中一种实现方式（基于字典的实现方式），实际操作中还有各种不一样的代码实现方式，这些会在最后统一介绍
#优化器是一类为了更好地管理神经网络参数更新策略的类
#随机梯度下降
#这是最基础的方法。每次用一小批数据算出梯度，然后沿梯度反方向更新参数
#优点在于简单且占用内存小
#缺点：对学习率敏感，容易收敛到局部最优而非全局最优
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, gradient):
        for key in params.keys():
            params[key] -= self.lr * gradient[f"d{key}"]
#动量优化器
#在SGD的基础上加上了惯性，它还累计了过去梯度的方向
#注重解决收敛到局部最优的问题，同时减少震荡
class Momentum:
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        #alpha是从上一步到下一步的速度保留量，用物理来直观地理解就是1-“阻力”
        self.alpha = alpha
        self.v = dict()

    def step(self, params, gradient):
        #创建速度矩阵，用来累累积过去梯度的下降方向
        if self.v == {}:
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.lr * gradient[f"d{key}"]
            params[key] += self.v[key]
#自适应梯度
#给每个参数单独配置学习率，历史梯度平方和大的参数，学习率自动便小
#注重解决特征稀疏时稀有参数更新少的问题（如 NLP 中低频词）
#但容易在后期出现不更新的情况
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = dict()

    def step(self, params, gradient):
        if self.h == {}:
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += gradient[f"d{key}"]**2
            params[key] -= self.lr * gradient[f"d{key}"] / np.sqrt(self.h[key] + 1e-7)
#指数移动平均
#自适应参数的改进版。它不再累积全部历史，而是改用指数移动平均只记录近期梯度平方
#注重解决原版学习率衰减过快的问题
#在非平稳目标下表现良好
class RMSprop:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.h = dict()

    def step(self, params, gradient):
        if self.h == {}:
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():#            这里的1-beta是保证只记录近期梯度平方的核心操作
            self.h[key] = self.beta*self.h[key] + (1-self.beta)*(gradient[f"d{key}"]**2)
            params[key] -= self.lr * gradient[f"d{key}"] / np.sqrt(self.h[key] + 1e-7)
#RMSprop和Momentum的结合版自适应学习率的随机梯度优化算法
#它结合了二者的优点。默认参数就能在多数任务上保持不错
#实现简单，收敛快，适合稀疏梯度和非平稳目标
#缺点在于泛化性能有时不足，它容易收敛到尖锐极小值
#它也有更先进的变体AdamW
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
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        for key in params.keys():
            #计算动量矩阵
            self.m[key] = self.beta_1*self.m[key] + (1-self.beta_1)*gradient[f"d{key}"]
            #计算速度平方均值
            self.v[key] = self.beta_2*self.v[key] + (1-self.beta_2)*(gradient[f"d{key}"]**2)
            #分别计算偏差修正的倍率
            #注意这里的一个设计：(1-self.beta_n**self.t)
            #由于m和v都初始化为零，而它们的更新是指数移动平均，就会使修正系数被(1-beta)缩小
            #例如第一步时的修正率就会是应有的0.1倍
            #所以要除以(1-self.beta_1**self.t)来恢复
            m_corrected = self.m[key] / (1-self.beta_1**self.t)
            v_corrected = self.v[key] / (1-self.beta_2**self.t)
            #参数更新
            params[key] -= self.lr*m_corrected / np.sqrt(v_corrected + 1e-7)

#接下来我们来介绍一些不同的编程实现思路，为了方便，这里全部使用SGD作为例子
#1.面向对象式，也是pytorch的实现风格
#定义一个类专门的类来存储参数
#接口清晰，梯度与参数绑定，避免字符串拼接和键名约定
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None

class ClassSGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

#2.函数式
#不修改原字典，而是返回更新后的新字典，适合函数式风格或需要保留历史参数的场景。
def sgd_step(params, gradient, lr=0.01):
    return {k: v - lr * gradient[f"d{k}"] for k, v in params.items()}
# 使用
#params = sgd_step(params, gradient, lr=0.1)

#3.基于参数名列表的显式映射
#不依赖 "d" + key 的隐式约定，而是显式传入参数名与梯度名的对应关系，或用结构化数据。
#命名约定可配置，灵活；避免硬编码字符串拼接。
# 多一层函数调用，略繁琐。
class ParameterSGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads, grad_key=lambda k: f"d{k}"):
        for key, value in params.items():
            value -= self.lr * grads[grad_key(key)]

#4. 用 dataclass 或 NamedTuple 组织参数与梯度
#把参数和梯度放进同一个结构化对象，减少两个字典不同步的风险。
#类型明确，参数与梯度成对出现，不易错配；适合层式网络结构。
#需要为每层定义结构，灵活性稍低。
from dataclasses import dataclass

@dataclass
class LayerParams:
    W: np.ndarray
    b: np.ndarray
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

class StructureSGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

#5.支持状态扩展的统一基类（为 Momentum/Adam 铺路）
#把“状态”抽象出来，SGD 只是无状态的特例，方便后续加动量、自适应学习率。
#模板方法模式，扩展新优化器只需重写 _update；状态管理集中。
#抽象层次多，小项目显得重。
class AbstractOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
        self.state = {}  # 子类可存速度、梯度平方等

    def step(self, gradient):
        for key, value in self.params.items():
            self._update(key, value, gradient[f"d{key}"])

    def _update(self, key, value, grad):
        value -= self.lr * grad   # 默认 SGD

class ParticularOptimizerMomentum(AbstractOptimizer):
    def __init__(self, params, lr=0.01, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta

    def _update(self, key, value, grad):
        v = self.state.get(key, 0)
        v = self.beta * v + grad
        self.state[key] = v
        value -= self.lr * v

#6.性能向：向量化 / 扁平化参数
#把参数字典展平成一个连续数组，梯度也展平，更新时只做一次向量运算，减少 Python 循环开销。
#循环少，适合参数多、单参数小的场景；便于与 C/GPU 后端对接。
#需要维护展平/还原映射，调试不直观。
class FlatSGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, flat_params, flat_grad):
        flat_params -= self.lr * flat_grad   # 单次向量化更新
#7.装饰器或回调式：在更新前后插入钩子
#适合需要日志、梯度裁剪、学习率调度等横切逻辑的场景。
#不侵入核心更新逻辑就能加功能（如梯度裁剪、记录范数）。
#钩子顺序和副作用需要小心管理。
class SGDWithHooks:
    def __init__(self, lr=0.01, hooks=None):
        self.lr = lr
        self.hooks = hooks or []

    def step(self, params, gradient):
        for hook in self.hooks:
            hook(params, gradient)
        for key in params:
            params[key] -= self.lr * gradient[f"d{key}"]
#核心权衡在于：参数与梯度如何组织（字典 vs 对象 vs 扁平数组）、是否有状态、更新是否原地。这三点决定了实现的风格和适用场景。