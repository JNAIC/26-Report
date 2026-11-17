from torch import nn
from torch.nn import Conv2d,MaxPool2d,ReLU,Linear,Flatten
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(1,16,3,1,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(16,32,3,1,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Flatten(),
            Linear(32*7*7,10),
            )

    def forward(self,x):
        return self.model(x)

if __name__ == "__main__":
    test_model = MyModule()
    test_data = torch.ones((64,1,28,28))
    test_output = test_model(test_data)
    print(f"输出形状是：{test_output.shape}")