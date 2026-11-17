# MoonShadow_week3


## 本周学习

> 这周基本都花在这一个程序上了，理论学习是囫囵吞枣般补完的，因为时间不够的原因，学习进度也严重滞后

### 实践操作/实践项目

#### 手写数字识别程序（基于 MNIST）

##### 1. 模型结构（`num_recog_model.py`）

模型本身写得比较简单，用 `nn.Sequential` 一个容器把层按顺序堆起来：

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(1, 16, 3, 1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, 3, 1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Linear(32*7*7, 10),
        )

    def forward(self, x):
        return self.model(x)
```

- 输入是 `(batch, 1, 28, 28)`：1 是单通道，28×28 是图片尺寸
- 卷积参数 `Conv2d(1, 16, 3, 1, padding=1)` 依次是：输入通道、输出通道、卷积核大小 3×3、步长 1、填充 1。**`padding=1` 是关键**，它让卷积之后尺寸还是 28×28 不变，否则每层会缩 2
- 池化 `MaxPool2d(2, 2)` 每次把尺寸减半，所以是 28 → 14 → 7
- 最后 `Linear(32*7*7, 10)`，1568 个特征接到 10 个类别上。**这个 7×7 是自己算出来的**：28 →（卷积不变）28 →（池化）14 →（卷积不变）14 →（池化）7，通道数 32，所以是 32×7×7 = 1568

> `torch.ones((64,1,28,28))` 是我从视频里看到的验证模型输出是否正确的方法，我靠这个走了一遍前向，打出来的输出形状是 `(64,10)`，说明模型输出是对的。

##### 2. 训练脚本（`手写数字识别.py`）

训练部分用的是最标准的四件套：`DataLoader` 分批次、`CrossEntropyLoss` 算损失、`SGD` 更新参数、`zero_grad/backward/step` 三步走。超参数：

- `batch_size`：训练 64，测试 32
- `learning_rate`：1e-2
- `epoch`：20

```python
for data in train_loader:
    imgs, targets = data
    outputs_train = model(imgs)
    loss_train = loss_fn(outputs_train, targets)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
```

> 由于是紧急搓出来的程序，时间吃紧，又还要写周报，所以我自己其实还没有完整跑过一遍这个模型。

所以大概就是这样，赶时间交差来的（
