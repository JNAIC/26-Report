import torch
import math
from torch._C import device
import numpy as np
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
loss_func = F.cross_entropy # 在这里负对数似然损失和对数softmax激活函数被pytorch结合为了一个cross_entropy
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
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

n,c = x_train.shape
lr = 0.1  # learning rate
epochs = 4  # how many epochs to train for

from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader

train_ds = TensorDataset(x_train,y_train)
valid_ds = TensorDataset( x_test , y_test )



class Mnist_logistic( nn.Module ):
    def __init__(self):
        super(Mnist_logistic, self).__init__()
        self.lin = nn.Linear(c, 10)

    def forward(self, xb):
        return self.lin(xb)
class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,stride= 2, padding=1 )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,stride= 2, padding=1 )
        self.conv3 = nn.Conv2d(32, 10, kernel_size=3, stride= 2, padding=1 )

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.adaptive_avg_pool2d(xb, 1)
        return xb.view(-1, xb.size(1))

def get_model():
    model = Mnist_CNN()
    model = model.to(device)
    return model, optim.SGD(model.parameters(), lr=lr,momentum=0.9)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    model.eval()
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = xb.to(device), yb.to(device)
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb)]
            )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print(0, val_loss)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb) ]
                )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch + 1, val_loss)
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

model, opt = get_model()

fit(epochs, model, loss_func, opt, train_dl, valid_dl)