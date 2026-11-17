
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # 第一层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二层
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = CNN()

    x = torch.randn(64, 1, 28, 28)

    output = model(x)

    print("输入形状：", x.shape)
    print("输出形状：", output.shape)
