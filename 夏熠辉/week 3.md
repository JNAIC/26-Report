# 手写数字识别

### 前置

* PyTorch是python调用的一个库,用做张量计算和训练神经网络

* MINIST是一个手写数字图片数据集

* 梯度下降权重计算

  ***

### 进程

#### 1.先读出一张MNIST图片和它的标签

在anaconda prompt运行

```python
conda activate pytorch
cd /d "C:\Users\z\Desktop\手写数字识别"
```

新建`inspect_data.py`写入

```python
from torchvision import datasets,transforms

train_data = datasets.MNIST(
	root = "data", #数据放在data文件夹内
	train = True,  #选择训练部分
	download = True, #缺少数据尝试自动下载
	transform = transforms.ToTensor( ),  #图片转化成张量
)

image,label = train_data[0] #取第一条数据

print("训练数据条数:", len(train_data))
print("第一张图片的形状:", image.shape)
print("第一张图片的标签:", label)
print("像素值范围:", image.min().item(), image.max().item())
```

运行`python inspect_data.py`

#### 2.划分数据

尝试把60000张图分成50000个训练图和10000个验证图

导入工具

```python
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
```

然后读取60000张图

```python
full_train_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)
```

和前面不同的是这里`download = Flase`表示直接读取本地数据,如果前面下载失败会报找不到数据集

接下来开始划分训练集和验证集

```python
train_data,val_data = random_split(
    full_train_data,
    [50000,10000],
    generator=torch.Generator().manual_seed(42)
)
```

> `train_data`训练更新模型参数,`val_data`检查模型表现,`random_split(要划分的数据集, 每部分的数量)`最后`torch.Generator()`是一个随机数生成器`manual_seed()`给它设置起始种子,这里选取42 ~~(原来42是个文化梗,我就说怎么种子都选的42)~~

设置每次取的图片(Dataloader)

```python
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True)
val_loader = DataLoader(
    val_data, 
    batch_size=64)
```

训练集中`shuffle`确定是否打乱顺序,`batch_size`表示最多取多少张图 

取出第一批图

```python
images, labels = next(iter(train_loader))
flat_images = images.flatten(start_dim=1)
```

> `iter(train_loader)` 得到一个可以逐批取数据的迭代器,`next(...)` 从这个迭代器取出第一批 `images, labels = ...` 拆成图片和标签,`images[0]` 是第一张图,`labels[0]` 是它的标签

> `flatten(start_dim = k)`,主要保留k左边的维度,把从k到末尾的维数合成一个,原代码start_dim = 1,[64,1,28,28]后面变成1 * 28 * 28 = 784,即[64,784] ~~(两个*没空格中间变斜体没绷住)~~

检查输出结果

```python
print("训练集和验证集大小", len(train_data), len(val_data))
print("一批图片的形状", images.shape)
print("一批标签的形状", labels.shape)
print("展平后的形状", flat_images.shape)
```

终端运行`python train.py`

#### 3.搭建网络,算出第一批预测

`train,py`的`import torch`下一行加上`from torch import nn`导入神经网络模块

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

> `nn.Sequential`按顺序执行各层,`nn.Flatten()`默认第一维开始展平
>
> Linear则根据128个不同的w,b将784个像素算出128个结果,第四步同理
>
> `nn.ReLU()`将得到的负值换为0

最后得到10个分数,找出最大分数

```python
images, labels = next(iter(train_loader))
logits = model(images)
predictions = logits.argmax(dim=1)
```

> 这里`argmax(dim = 1)`意思是从从第一维开始找

然后是计算误差

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

> 第一行是一种扣分规则(交叉熵),第二行是扣分计算

检查结果

```py
print("模型输出形状", logits.shape)
print("第一张图的预测", predictions[0].item())
print("第一张图的真实标签", labels[0].item())
print("这一批的损失", loss.item())
```

> 我的结果:
>
> 模型输出形状 torch.Size([64, 10])
> 第一张图的预测 5
> 第一张图的真实标签 3
> 这一批的损失 2.313380002975464

#### 4.第一轮训练

训练开始前,准备好分组和全连接网络

然后就是选设备

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

> `model.to(device)`把权重偏移到指定设备上
>
> 第四行则是权限移交给Adam,参数控制,lr是学习率

准备进入函数,准备好相关信息交进去

```py
print("训练设备", device, flush=True) #显示设备信息
train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
```

函数部分

```py
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
```

用得到的数据更新参数(续上def)

```py
optimizer.zero_grad() #清除上次留的梯度
logits = model(images) 
loss = loss_fn(logits, labels) #平均损失
loss.backward() #梯度
optimizer.step() #由梯度改参数
```

> `model(images)` 算分,此时`logits` 形状[64,10],每一行是一张图对应的十个分数

记录这一批的损失,求一整轮的平均

```py
batch_size = labels.size(0) #这批实际图数
total_loss += loss.item() * batch_size
total_samples += batch_size
```

最后返回平均损失

```py
return total_loss / total_samples
```

ok,这样就算成功完成了一轮

> 我的输出:
>
> 训练设备 cuda
> 第 1 轮平均训练损失 0.3709



#### 5.计算验证准确率

添加验证函数

```py
@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval() #运行模式
    correct = 0 #正确数
    total = 0 #总数
    for images, labels in loader:
    	images = images.to(device)
    	labels = labels.to(device)

    	logits = model(images)
    	predictions = logits.argmax(dim=1)
    	correct += (predictions == labels).sum().item()
    	total += labels.size(0)
```

> `.sum()`把`True`当作1,`Flase` 当作0

外层循环跑5轮

```py
train_losses = []
val_accuracies = []

for epoch in range(1, 6):
    train_loss = train_one_epoch(
        model, train_loader, loss_fn, optimizer, device
    )
    val_accuracy = evaluate_accuracy(model, val_loader, device)
    
    train_losses.append(train_loss)
	val_accuracies.append(val_accuracy)
```

打印结果

```py
print(
    f"第 {epoch} 轮 | "
    f"平均训练损失 {train_loss:.4f} | "
    f"验证准确率 {val_accuracy:.2%}",
    flush=True,
)
```

**训练损失看的是模型在训练过程中给正确答案打分的情况，验证准确率看的是这一轮训练结束后，模型在验证图上有多少次猜对**(误区2:过拟合)

这里还有验证集的取出代码差点忘了

```py
val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle = False
)
```

> 我的训练结果:
>
> 第 1 轮 | 平均训练损失 0.3800 | 验证准确率 93.33%
> 第 2 轮 | 平均训练损失 0.1784 | 验证准确率 95.19%
> 第 3 轮 | 平均训练损失 0.1268 | 验证准确率 96.14%
> 第 4 轮 | 平均训练损失 0.0968 | 验证准确率 96.40%
> 第 5 轮 | 平均训练损失 0.0758 | 验证准确率 96.91%

#### 6.cnn

这是单通道

```py
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2),
```

然后把不同通道特征组合起来,`in_channel = 16`

```python
nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2),
```

把特征变成分类分数

```python
nn.Flatten(),
nn.Linear(32 * 7 * 7, 10),
```

再次训练

> 我的结果:
>
> 第 1 轮 | 平均训练损失 0.2625 | 验证准确率 97.07%
> 第 2 轮 | 平均训练损失 0.0759 | 验证准确率 97.57%
> 第 3 轮 | 平均训练损失 0.0565 | 验证准确率 98.32%
> 第 4 轮 | 平均训练损失 0.0452 | 验证准确率 98.00%
> 第 5 轮 | 平均训练损失 0.0379 | 验证准确率 98.54%

#### 7.学习率对照

这次换不同的学习率,在此之前先先固定模型的初始权重`torch.manual_seed(42)`

然后做两轮实验,学习率分别为lr = 0.001,lr = 0.002

> lr = 0.001
>
> 第 1 轮 | 平均训练损失 0.2872 | 验证准确率 96.22%
> 第 2 轮 | 平均训练损失 0.0834 | 验证准确率 97.78%
> 第 3 轮 | 平均训练损失 0.0605 | 验证准确率 98.07%
> 第 4 轮 | 平均训练损失 0.0490 | 验证准确率 97.85%
> 第 5 轮 | 平均训练损失 0.0411 | 验证准确率 98.20%
>
> lr = 0.002
>
> 第 1 轮 | 平均训练损失 0.2137 | 验证准确率 97.02%
> 第 2 轮 | 平均训练损失 0.0655 | 验证准确率 98.07%
> 第 3 轮 | 平均训练损失 0.0465 | 验证准确率 98.24%
> 第 4 轮 | 平均训练损失 0.0387 | 验证准确率 98.00%
> 第 5 轮 | 平均训练损失 0.0318 | 验证准确率 98.35%

#### 8.训练损失和验证准确率曲线

这个我绘图没有去学习了,数据来源于7中的训练数据

![image-20261113172635154](C:\Users\z\AppData\Roaming\Typora\typora-user-images\image-20261113172635154.png)

#### 9.测试集评估

读取官方测试集,测试图放入模型

```py
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms.ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=64)

test_accuracy = evaluate_accuracy(model, test_loader, device)
print(f"测试样本数 {len(test_data)} | 测试准确率 {test_accuracy:.2%}")
```

这是全连接网络(lr=0.001)和卷积神经网络(lr=0.002)测试结果

![image-20261113174539602](C:\Users\z\AppData\Roaming\Typora\typora-user-images\image-20261113174539602.png)

