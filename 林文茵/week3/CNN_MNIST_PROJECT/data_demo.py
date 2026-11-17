import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

transform = transforms.ToTensor() #将图片转换为张量Tensor

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)
print("训试集数量：",len(train_dataset))
print("测试集数量：",len(test_dataset))

image,label = train_dataset[0]

print("图片张量形状：",image.shape)
print("数字标签：",label)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

print("图片的像素矩阵：")
print(image[0])

print("最大像素值：", image.max())
print("最小像素值：", image.min())

images, labels = next(iter(train_loader))

print("一批图片的形状：", images.shape)
print("一批标签的形状：", labels.shape)