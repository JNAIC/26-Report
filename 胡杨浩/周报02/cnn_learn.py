import torch
import math
from torch._C import device
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from pathlib import Path # pathlib是python自带的文件处理模块
data_path = Path("../data")
path = data_path / "mnist" # “/”是一个被重载了的运算符，用于将string直接与path链接为新的path对象
path.mkdir(parents=True,exist_ok=True)
filename = "mnist.pkl.gz"
import pickle
import gzip
with gzip.open((path/filename).as_posix(),"rb") as f:
    ((x_train, y_train), (x_test, y_test),_) = pickle.load(f,encoding="latin-1")
#gzip是用于处理.zp文件的模块，pickle用于将二进制文件转换为python的类型，如numpy，string
import torch
#将numpy转为tensor格式
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)
train_ds = TensorDataset(x_train,y_train)
valid_ds = TensorDataset( x_test , y_test )
print("Data loaded")
#以上都是文件处理

# 接下来要构建的是一个卷积神经网络
# 首先我们需要了解何为卷积
# 形式上来说，卷积会使用一个卷积核逐步地与数据集中的每一项相乘，从而达到提取某种数据特征的目的
# 我们来看一个最简单的例子：
# 假设现在我们有一个由10个数构成的集合
# members = [3,6,0,9,3,17,5,3,5,3]
# R_core = [0.2,0.2,0.2,0.2,0.2] # 同时有一个由五个数构成的卷积核
# result = []
# #现对其进行卷积操作：
# print(len(members))
# for i in range(0,len(members) - len(R_core) + 1):
#     m_res = 0
#     n = 0
#     while n < len(R_core):
#         res = members[n+i] * R_core[n]
#         m_res += res
#         n += 1
#     result.append(m_res)
#     i += 1
#     n = 0
# print(result)
#示例输出：
# 10
# [4.200000000000001, 7.0, 6.800000000000001, 7.4, 6.6, 6.6]
# 示例中展示的卷积操作相当于在求数据集的局部平均值，例如result的第2项为7.0，对应members中6，0，9，3，17五个数的平均值
# 同时，通过修改卷积核，我们可以提取出更加丰富且多样化的数据特征
# 例如在图像识别工作中，我们可以构建一个这样的卷积核：
# [[1  ,1  ,1  ],
#  [0.5,0.5,0.5],
#  [0  ,0,  0  ]]
# 像这个卷积核就可以检查某个地方是否有水平的边，当被卷积区域存在水平的边时，卷积的输出值的绝对值就会偏大
# 那么接下来，为了处理MNIST数据集中的28*28手写数字灰度图像，我们将要构建下面这个卷积神经网络：
#



import torch.nn.functional as F
# torch.nn.functional是torch中一个类似于函数工具箱的类，里面有神经网络训练常用的各种函数
loss_func = F.cross_entropy # 在这里负对数似然损失和对数softmax归一化函数被pytorch结合为了一个cross_entropy
# loss_batch函数，用于计算损失，可以使用optimizer对象，也可以不使用
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    #如果使用了优化器对象，那么就会进行反向传播，也就是训练，没有使用opt时，此函数就可以拿来当作损失函数使用
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb) # 返回值是一个float和int,分别是平均损失和长度，
    # item()函数用来把loss_func返回的零维张量改回float方便numpy计算
    # 将cuda.0上的四字节考回主机内存，同时去掉autograd

bs = 32 # batch size

def accuracy(out,yb):
    preds = torch.argmax( out , dim = 1 )
    return ( preds == yb ).float().mean()
# 准确率函数，这里out接受的是一个bs*10的tensor，
# argmax函数在结果中按照序号为1的维度寻找最大值，体现为在shape属性中索引为一的维度，也就是在十个值中寻找最大值
# preds会是一个bs*1的tensor
# 返回值中，preds == yb 是对tensor的逻辑运算，得到一个shape与preds相同的tensor，dtype=bool
# 将其数据类型转换为float后，只剩下1.0与0.0，这时再求和即得准确率

#负对数似然损失函数：
# 它将结果沿最后一个维度(我们可以在gey_data()函数里得到x的形状和每个维度对应的形状，这里是batch_size的维度)求和并取对数后减原数据。
# 它的作用是计算损失，它的后一项就是常用的softmax归一化函数
# 不过在本脚本中，它并没有被直接使用，loss_func使用了torch.nn.functional中的cross_entropy函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1) # unsqueeze()的作用是在最后重新插入一个维度，因为sum()求和后tensor会少一个维度

#这个是线性神经网络，在本例中未曾使用
class Mnist_logistic( nn.Module ):
    def __init__(self):
        super(Mnist_logistic, self).__init__()
        self.lin = nn.Linear(c, 10)

    def forward(self, xb):
        return self.lin(xb)
# 这个是我们的重头戏：卷积神经网络。
class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__() # super()函数
        # channels通道数，直观理解就是在这张图片上的某一个像素点由几个数描述
        # 例如conv1层，它直接接收初始数据，而MNIST数据集中的图片是灰度图，仅由一个数字来描述灰度，所以channels=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,stride= 2, padding=1 )
        # 接下来，对于第二层就不一样了。
        # 在第一层中我们已经设定了输出通道数为16，torch会为我们自动生成16个卷积核，每个卷积核的形状为3*3，共9*16=144个参数，未设置bias=false,还有16个偏置参数，第一层共计160个参数
        # 同时第一层的stride为2，每卷积一次，卷积核做下一次卷积操作时在图片上的位置就移动两个位置
        # padding填充，用于在输入图像的边缘补充像素，通常补0，这里也一样。它的意义是防止边缘信息丢失，如果不填充，那么图像边缘的像素只会被卷积核扫到一次。
        # 同时它也可以在stride=1时通过恰当的填充使输入与输出的长宽完全一致。但是在这里，我们没有这个需求，对3*3的卷积核填充1即可。
        # 输入是一个shape为(32（batch_size）,1（通道数）,28,28)的tensor，dtype为int8
        # 输出的shape为(32,16,28（图像尺寸）-3（卷积核尺寸）+2（填充量*2）)/2（步长） + 1（植树原理，卷积操作在图像两侧都要做，所以加一）= 14.5 -> 14（向下取整）,14)
        # 这时conv2层的卷积核的shape就会是(16,3,3)，共有输出通道数32个卷积核，该层总参数量为3*3*16*32 + 32（偏置量）= 4640
        self.conv2 = nn.Conv2d(16 , 32 , kernel_size=3 , stride= 2 , padding=1 )
        self.conv3 = nn.Conv2d(32 , 10 , kernel_size=3 , stride= 2 , padding=1 )

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28) # 这里用view函数将MNIST数据集原有的二维tensor(32,784)转变为(32,1,28,28)从而方便做卷积操作
        # view的四个参数的意义是：-1：自动推断，传进来的tensor的第一个维度是多长它就填多长，这里是batch_size，同时也可以在最后一个batch（50000%32 = 8）时保证安全
        # 1：第一层卷积层的通道数 (28,28):图像尺寸

        xb = F.relu(self.conv1(xb))#前向过程中依次通过三个卷积层同时使用了经典激活函数relu（max(0,x)）
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.adaptive_avg_pool2d(xb, 1)#平均池化函数
        # 对池化函数的解释要联系第三个卷积层，conv3输出(32,10,4,4)的语义是，10个通道=10个类别，4*4个位置是在说这张图上的16个局部区域对数字x的响应
        # 平均池化层对数字x的响应求平均来对答案打分
        return xb.view(-1, xb.size(1))#重新调整输出张量shape为(32,10)方便损失函数使用
# get_model函数
def get_model():
    model = Mnist_CNN()
    model = model.to(device)
    return model, optim.SGD(model.parameters(), lr=lr,momentum=0.9)
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

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    # 计算未训练前model的表现，方便观察训练效果
    model.eval()
    with torch.no_grad():#上下文管理器
        # no_grad可以帮自动我们开关建图过程
        # 因为测试过程不需要做反向传播，也就不需要autograd，ReLU 的 mask、卷积的 im2col 缓冲等

        # 把各batch的loss按样本数加权平均，得到整个验证集的平均损失
        losses, nums = zip(*[
            # 这里每一轮产出的是一个二元组
            # zip()函数是转置技巧 —— 把"每批一行"的表变成"每列一个元组"：
            # 输入列表:  [(0.4, 64), (0.6, 64), (0.9, 16)]
            #               ↓ zip(* ) 转置
            # losses:    (0.4,      0.6,      0.9 )
            # nums:      (64,       64,       16  )
            loss_batch(model, loss_func, xb.to(device), yb.to(device)) #这里没有使用optimizer，因为我们只想做测试
            for xb, yb in valid_dl #使用dataloader来获取数据，同时记得这里的xb是一个shape为(32,784)的tensor，而yb为(1,32)
        ])
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums) #计算加权平均loss
    print(0, val_loss)#输出
    for epoch in range(epochs): #在整个训练集上训练epochs次，每个epoch都要在训练集上跑一次，在测试集上跑一次
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)#to(device)将数据搬到GPU上
            loss_batch(model, loss_func, xb, yb, opt)  #对比前面，我们这里向loss_batch()中传入了优化器对象
        #与上面一样的评估流程，也可以将其封装为函数
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[
                loss_batch(model, loss_func, xb.to(device), yb.to(device))
                for xb, yb in valid_dl
            ])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch + 1, val_loss)

# 用于获取数据的函数，它在这里所接受的是我们要用于训练和测试的数据，然后返回两个dataloader
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True), # 这里我们为dataloader设定了参数，使用train_ds作为训练集，batch_size，提取数据时会自动打乱
        DataLoader(valid_ds, batch_size=bs * 2),
    )
#训练用的超参数
n,c = x_train.shape
lr = 0.1  # learning rate
epochs = 4  # how many epochs to train for
#文件处理部分在顶部已经完成，中间部分我们一直在定义各种函数和类，也就是制造我们要使用的工具
#由于我们的准备工作足够充分，我们真正的训练代码可以用三行来完成
#构建dataloader
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
#构建model和optimizer
model, opt = get_model()
#调用训练函数
fit(epochs, model, loss_func, opt, train_dl, valid_dl)