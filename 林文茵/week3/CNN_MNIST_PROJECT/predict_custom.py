
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CNN().to(device)

model.load_state_dict(
    torch.load("./models/mnist_cnn.pth", map_location=device)
)

model.eval()


# 输入图片路径
image_path = input("请输入图片路径：")

# 读取图片并转换为灰度图
image = Image.open(image_path).convert("L")

# 将白底黑字转换为黑底白字
image = ImageOps.invert(image)

# 调整图片大小为 28×28
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0).to(device)


# 进行预测
with torch.no_grad():
    output = model(image)
    predicted = torch.argmax(output, dim=1).item()


# 输出结果
print("模型预测的数字是：", predicted)
