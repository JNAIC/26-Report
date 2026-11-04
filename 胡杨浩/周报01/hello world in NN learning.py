import torch
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
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from pathlib import Path
import  requests
data_path = Path("data")
path = data_path / "mnist"

path.mkdir(parents=True,exist_ok=True)

url = "https://github.com/pytorch/tutorials/raw/main/_static"
filename = "mnist.pkl.gz"
import pickle
import gzip
print(1)

with gzip.open((path/filename).as_posix(),"rb") as f:
    ((x_train, y_train), (x_test, y_test),_) = pickle.load(f,encoding="latin-1")
import torch

#x_train,y_train,x_test,y_test = map(torch.tensor, (x_train,y_train,x_test,y_test))
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

import math
weights = (torch.randn(784,10) / math.sqrt(784)).to(device)
weights.requires_grad = True
print(weights.grad)
bias = torch.zeros(10).to(device)
bias.requires_grad = True
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb,weights,biases):
    result = log_softmax(xb @ weights + biases)
    return result

bs = 256
xb = x_train[0:bs].to(device)
preds = model(xb,weights,bias).to(device)
print(preds[0], preds.shape)
print(bias.grad)
def nll(input,target):
    return -input[range(target.shape[0]),target].mean()

loss_func = nll

yb = y_train[0:bs].to(device)
print(loss_func(preds,yb))

def accuracy(out,yb):
    preds = torch.argmax(out,dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds,yb))

n,c = x_train.shape
print(n)
lr = 2.0  # learning rate
epochs = 2  # how many epochs to train for
print(loss_func(model(xb,weights,bias), yb), accuracy(model(xb,weights,bias), yb))
for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb,weights,bias)
        loss = loss_func(pred, yb).to(device)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
    print(loss_func(pred, yb), accuracy(pred, yb))

