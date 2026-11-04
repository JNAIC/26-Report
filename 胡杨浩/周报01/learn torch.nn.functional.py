import torch
import math
from torch._C import device
# 接下来构建的模型是这样的
###
# 首先，它是一个没有隐藏层只有输入输出层的模型
# 输入784个int8整型
# 输出10个float32
# 以输出中最大的一个值的索引作为对MNIST数据集内一张图片的分类结果
# 现定义以下变量及其意义：
# x_train 训练集
# y_train 训练集答案
# weights 一个784*10的矩阵与输入的784维向量作矩阵乘法得到一个1*10的矩阵作为输出
# bias 偏离量 一个1*10的矩阵，与输出相加得到结果
# 主要函数
# model：用于将input与weights作矩阵乘法并加上bias，input中有batch size个元素
# log_softmax:用于将model的输出中的元素分别作为e^x的指数并在求和后作以e为底的指数运算
# ###

#
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.version.cuda)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
from pathlib import Path # pathlib是python自带的文件处理模块
data_path = Path("data")
path = data_path / "mnist" # “/”是一个被重载了的运算符，用于将string直接与path链接为新的path对象
path.mkdir(parents=True,exist_ok=True)
filename = "mnist.pkl.gz"
import pickle
import gzip
print(1)
with gzip.open((path/filename).as_posix(),"rb") as f:
    ((x_train, y_train), (x_test, y_test),_) = pickle.load(f,encoding="latin-1")
#gzip是用于处理.zp文件的模块，pickle用于将二进制文件转换为python的类型，如numpy，string
import torch
#将numpy转为tensor格式
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
#以上都是文件处理
import torch.nn.functional as F
#torch.nn.functional模块包含了torch.nn中所有的函数
# 包括广泛的损失和激活函数，池化函数等。库中也有一些进行卷积、线性层等操作的函数
loss_func = F.cross_entropy # 在这里负对数似然损失和对数softmax激活函数被pytorch结合为了一个cross_entropy


bs = 32 # batch size
def accuracy(out,yb):
    preds = torch.argmax( out , dim = 1 )
    return ( preds == yb ).float().mean()
# 准确率函数，这里out接受的是一个bs*10的tensor，
# argmax函数在结果中按照序号为1的维度寻找最大值，体现为在shape属性中索引为一的维度，也就是在十个值中寻找最大值
# preds会是一个bs*1的tensor
# 返回值中，preds == yb 是对tensor的逻辑运算，得到一个shape与preds相同的tensor，dtype=bool
# 将其数据类型转换为float后，只剩下1.0与0.0，这时再求和即得准确率
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)
# 负对数似然损失函数，接受一个bs*10的tensor。
#x.exp()对tensor中的每一项n求e^n
# sum(-1)对最后一个维度求和这里只有两个维度：bs与类别
# log()再求以e为底的对数
# 此时x的shape为bs*10，但是运算结果的shape为bs,需要unsqueeze来将其恢复为bs*1
# tensor([[0, 1, 0],
#         [0, 0, 1],
#         [0, 1, 0],
#         [0, 2, 0]])    不同形状的tensor做减法效果如图
# tensor([[1.5514],
#         [1.5514],
#         [1.5514],
#         [2.2395]])
# tensor([[-1.5514, -0.5514, -1.5514],
#         [-1.5514, -1.5514, -0.5514],
#         [-1.5514, -0.5514, -1.5514],
#         [-2.2395, -0.2395, -2.2395]])
n,c = x_train.shape
lr = 0.01  # learning rate
epochs = 4  # how many epochs to train for

from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
# dataset与Dataloader是torch提供的用于加载数据的类
# Dataset 可以是任何具有 __len__ 函数（由 Python 的标准 len 函数调用）和 __getitem__ 函数作为索引方式的对象。
#PyTorch 的 TensorDataset 是一个封装张量的 Dataset。通过定义长度和索引方式，这也为我们提供了一种在沿张量的第一维迭代、索引和切片方面的便利方法。这将在我们训练时更容易同时访问独立变量和依赖变量。
#PyTorch 的 DataLoader 负责管理批次
#可以从任意 Dataset 创建一个 DataLoader。DataLoader 使我们更容易地迭代批次。
#DataLoader 会自动为我们提供每个 minibatch。
train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader( train_ds , batch_size=bs , shuffle=True )
valid_ds = TensorDataset( x_test , y_test )
valid_dl = DataLoader( valid_ds , batch_size=bs * 2 )
# 接下来，我们将使用nn.Module和nn.Parameter
#我们需要创建一个类来保存我们的权重、偏置以及前向步骤的方法。nn.Module具有许多属性和方法（例如.parameters()和.zero_grad()），我们将使用它们。
#
#既然我们现在使用的是一个对象而不是仅仅使用一个函数，首先我们需要实例化我们的模型：
#nn.Module 对象被用作函数（即它们是 可调用的），在调用nn.Module对象时，Pytorch 会自动调用我们的 forward 方法。
#现在我们可以利用 model.parameters() 和 model.zero_grad()（这两个方法均由 PyTorch 定义，如 nn.Module 所示）来使这些步骤更加简洁，并且减少忘记某些参数的风险
#要想替代手工地初始化weights和bias，可以使用Pytorch类 nn.Linear 来创建一个线性层
#Pytorch有许多预定义的层类型，可以大大简化我们的代码，并且通常也会使其运行得更快。
#

class Mnist_logistic( nn.Module ):
    def __init__(self):
        super(Mnist_logistic, self).__init__()
        self.lin = nn.Linear(c, 10)

    def forward(self, xb):
        return self.lin(xb)
def get_model():
    model = Mnist_logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
model, opt = get_model()
# 这里的 opt 是优化器（optimizer）对象，具体来说是 torch.optim.SGD 的一个实例。
#optim.SGD	随机梯度下降优化器类（torch.optim.SGD）
# model.parameters()	模型所有可训练参数（权重、偏置），即需要更新的张量
# lr=lr	学习率（learning rate）
# 返回值 opt	一个优化器对象，负责根据梯度更新这些参数#
#要更新的参数（model.parameters() 的引用）
# 更新规则和超参数（SGD 的 lr、momentum 等）
# 常用属性和方法
# opt.param_groups          # 参数组列表（含 lr 等超参）
# opt.param_groups[0]['lr'] # 查看当前学习率
# opt.state_dict()          # 保存优化器状态（如 momentum buffer）
# opt.load_state_dict(...)  # 恢复状态（断点续训）
# opt.zero_grad()           # 清零梯度
# opt.step()                # 更新参数
def fit():
    for epoch in range(epochs):
        model.train() # 切换状态，这时会有两个主要效果，一是模型会随机地丢弃一部分参数以防止过拟合
        for xb,yb in train_dl:
            pred = model(xb)  # 前向
            loss = loss_func(pred, yb) # 计算损失
            loss.backward() # 清梯度
            opt.step() # 反向传播，算梯度
            opt.zero_grad() # 用 opt 更新参数 ← 关键
        print("epoch:",epoch + 1,"loss:",loss_func(pred, yb),"acc:",accuracy(pred, yb))
        model.eval() # 状态切换为测试
        with torch.no_grad(): # with上下文管理器
            valid_loss = sum(loss_func(model(xb),yb) for xb , yb in valid_dl)
            print("valid loss:",valid_loss/len(valid_dl))
fit()
