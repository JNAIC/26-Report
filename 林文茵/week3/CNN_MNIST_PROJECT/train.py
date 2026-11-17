
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("当前使用设备：", device)

if torch.cuda.is_available():
    print("GPU名称：", torch.cuda.get_device_name(0))
else:
    print("未检测到CUDA，请检查PyTorch环境。")


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)


model = CNN().to(device)


# 定义损失函数及优化器
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# 5. 开始训练
epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算本轮平均损失
    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# 保存模型
os.makedirs("./models", exist_ok=True)

torch.save(model.state_dict(), "./models/mnist_cnn.pth")

print("模型训练完成！")
print("模型已保存到 ./models/mnist_cnn.pth")
