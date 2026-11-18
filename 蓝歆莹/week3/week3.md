# 周报

## 一、概况

在AI的帮助下大致完成了这个手写数字识别项目，虽然对cnn的理解加深了，但好多代码其实还看不太懂，感觉直接让AI给的代码太复杂了，之后再尝试独立完成保留主要内容且简单一些的完整代码。

## 二、对项目的基本理解

把一张手写数字图片作为像素矩阵输入cnn。卷积层提取局部特征，ReLU引入非线性，池化缩小特征图，全连接层综合这些特征，输出0～9的类别分数。训练时衡量预测与标签的差距，通过反向传播计算梯度，再由优化器更新参数；训练完成后，用未参与训练的图片测试识别效果。

## 三、训练设置

- 训练集：50,000 张；验证集：10,000 张；测试集：10,000 张。
- 将官方 60,000 张训练图片随机划分为训练集和验证集，使用固定随机种子 42，保存划分索引。
- 图片转换：使用`ToTensor()`，像素值缩放到 0～1。
- 批大小：64；训练轮数：3；优化器：Adam；学习率：0.001。
- 损失函数：交叉熵；运行设备：CPU。

## 四、代码与说明
大致能理解的一些关键代码

### 1. 数据转换与批量读取

```python
transform = transforms.ToTensor()

train_loader = DataLoader(
    train,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
    num_workers=0
)
```

`ToTensor()` 把图片转换成张量，并将像素值缩放到 0～1。`DataLoader` 每次取出一批图片和对应标签，`shuffle=True` 表示训练时打乱取样顺序。通常一批图片的形状是 `[64, 1, 28, 28]`。

### 2. CNN 的特征提取部分

```python
self.features = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```

第一层卷积把 1 个灰度通道变成 16 个特征通道，第二层把 16 个通道变成 32 个通道。两次池化使图片高和宽从 28 缩小到 14，再缩小到 7。


### 3. 分类部分与前向传播

```python
self.classifier = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

def forward(self, images):
    features = self.features(images)
    return self.classifier(features)
```

`Flatten(start_dim=1)` 保留批次维度，把每张图片的 `32×7×7` 个特征展平成 1568 个数。全连接层再把这些特征综合成十个类别分数。

`forward()` 规定数据的流向。调用 `model(images)` 时，会自动执行这个过程。输出是原始分数，称为 logits；由于交叉熵损失直接接收这些分数，输出层没有额外添加 Softmax。

### 4. 损失函数与优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
```

`criterion` 负责对照预测与标签计算损失。`optimizer` 负责调整模型中的可学习参数，`args.lr` 本次取值为 0.001。

### 5. 一批数据的训练过程

```python
model.train()

for batch_index, (images, labels) in enumerate(loader, start=1):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

1. `model.train()`：切换到训练模式。
2. `.to(device)`：让模型使用的数据放在相同的计算设备上。
3. `optimizer.zero_grad()`：清空旧梯度，模型权重仍然保留。
4. `model(images)`：输入图片，得到模型预测分数。
5. `criterion(logits, labels)`：用正确标签计算预测损失。
6. `loss.backward()`：计算各个参数的梯度，此时尚未更新参数。
7. `optimizer.step()`：根据梯度实际更新参数。

## 五、实际运行结果

### 1. 每轮训练与验证结果

| 轮次 | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 |
|---|---:|---:|---:|---:|
| 1 | 0.2750 | 91.55% | 0.0973 | 97.17% |
| 2 | 0.0715 | 97.84% | 0.0635 | 98.10% |
| 3 | 0.0461 | 98.58% | 0.0553 | 98.35% |

### 2. 最终测试结果

根据验证准确率选出的最佳模型来自第 3 轮。在 10,000 张独立测试图片上：

- 测试准确率：**98.65%**。
- 识别正确：**9,865 张**。
- 识别错误：**135 张**。
- 测试损失：约 **0.0421**。
