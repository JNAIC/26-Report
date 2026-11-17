import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random

from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

model = CNN().to(device)

model.load_state_dict(
    torch.load("./models/mnist_cnn.pth", map_location=device)
)

model.eval()


# 随机选择一张图片
index = random.randint(0, len(test_dataset) - 1)

image, label = test_dataset[index]

# 增加一个 batch 维度
image_input = image.unsqueeze(0).to(device)


# 模型预测
with torch.no_grad():
    output = model(image_input)
    predicted = torch.argmax(output, dim=1).item()


print("真实数字：", label)
print("预测数字：", predicted)

if predicted == label:
    print("预测正确！")
else:
    print("预测错误！")

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True: {label}, Predicted: {predicted}")
plt.axis("off")
plt.show()
