
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("当前使用设备：", device)


transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)


# 加载模型
model = CNN().to(device)

model.load_state_dict(
    torch.load("./models/mnist_cnn.pth", map_location=device)
)

model.eval()


# 测试
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # 获取预测概率最大的数字
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


# 输出准确率
accuracy = 100 * correct / total

print("测试图片总数：", total)
print("识别正确数量：", correct)
print(f"测试集准确率：{accuracy:.2f}%")
