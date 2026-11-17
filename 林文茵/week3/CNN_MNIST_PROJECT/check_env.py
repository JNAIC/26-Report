import torch
import torchvision
import matplotlib

print("PyTorch版本：", torch.__version__)
print("Torchvision版本：", torchvision.__version__)
print("Matplotlib版本：", matplotlib.__version__)

print("CUDA是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU型号：", torch.cuda.get_device_name(0))
    x = torch.randn(3, 3).cuda()
    print("GPU测试成功！")
    print(x)
else:
    print("当前使用CPU运行")