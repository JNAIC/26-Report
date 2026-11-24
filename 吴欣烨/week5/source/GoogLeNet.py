import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    download=True, 
    train=True, 
    transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    download=True, 
    train=False, 
    transform=transform_test
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

class Branch_maxpool_conv(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                maxpool_kernel_size: int, 
                maxpool_stride: int, 
                maxpool_padding: int, 
                conv_kernel_size: int, 
                conv_stride: int, 
                conv_padding: int, 
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=maxpool_kernel_size, 
                stride=maxpool_stride, 
                padding=maxpool_padding
            ), 
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=conv_kernel_size, 
                stride=conv_stride, 
                padding=conv_padding, 
                bias=False
            ), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.branch(input)

class Branch_conv_conv(nn.Module):
    def __init__(self, 
                    in_channels: int, 
                    hidden_channels: int, 
                    out_channels: int, 
                    conv1_kernel_size: int, 
                    conv1_stride: int, 
                    conv1_padding: int, 
                    conv2_kernel_size: int, 
                    conv2_stride: int, 
                    conv2_padding: int, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=hidden_channels, 
                kernel_size=conv1_kernel_size, 
                stride=conv1_stride, 
                padding=conv1_padding, 
                bias=False
            ), 
            nn.BatchNorm2d(hidden_channels), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(
                in_channels=hidden_channels, 
                out_channels=out_channels, 
                kernel_size=conv2_kernel_size, 
                stride=conv2_stride, 
                padding=conv2_padding, 
                bias=False
            ), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.branch(input)

class Branch_conv(nn.Module):
    def __init__(self, 
                    in_channels: int, 
                    out_channels: int, 
                    kernel_size: int, 
                    stride: int, 
                    padding: int, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=False
            ), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.branch(input)

class Inception(nn.Module):
    def __init__(self,
                 branches: list[Branch_maxpool_conv | Branch_conv_conv | Branch_conv], 
                  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branches = nn.ModuleList(branches)

    def forward(self, input):
        results = [branch(input) for branch in self.branches]
        return torch.cat(results, dim=1)
        

class GoogLeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=32, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False
            ), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 

            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False
            ), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(
                kernel_size=3, 
                stride=2, 
                padding=1
            )
        )

        self.inception_1 = Inception(
            branches=[
                Branch_maxpool_conv(        # Branch 1
                    in_channels=64, 
                    out_channels=16, 
                    maxpool_kernel_size=3, 
                    maxpool_stride=1, 
                    maxpool_padding=1, 
                    conv_kernel_size=1, 
                    conv_stride=1, 
                    conv_padding=0
                ), 
                Branch_conv_conv(           # Branch 2
                    in_channels=64, 
                    hidden_channels=32, 
                    out_channels=32, 
                    conv1_kernel_size=1, 
                    conv1_stride=1, 
                    conv1_padding=0, 
                    conv2_kernel_size=3, 
                    conv2_stride=1, 
                    conv2_padding=1
                ), 
                Branch_conv_conv(           # Branch 3
                    in_channels=64, 
                    hidden_channels=32, 
                    out_channels=32, 
                    conv1_kernel_size=1, 
                    conv1_stride=1, 
                    conv1_padding=0, 
                    conv2_kernel_size=5, 
                    conv2_stride=1, 
                    conv2_padding=2
                ), 
                Branch_conv(                # Branch 4
                    in_channels=64, 
                    out_channels=16, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0
                )
            ]
        )   # out_channels=96

        self.inception_2 = nn.Sequential(
            Inception(
                branches=[
                    Branch_maxpool_conv(        # Branch 1
                        in_channels=96, 
                        out_channels=16, 
                        maxpool_kernel_size=3, 
                        maxpool_stride=1, 
                        maxpool_padding=1, 
                        conv_kernel_size=1, 
                        conv_stride=1, 
                        conv_padding=0
                    ), 
                    Branch_conv_conv(           # Branch 2
                        in_channels=96, 
                        hidden_channels=16, 
                        out_channels=64, 
                        conv1_kernel_size=1, 
                        conv1_stride=1, 
                        conv1_padding=0, 
                        conv2_kernel_size=3, 
                        conv2_stride=1, 
                        conv2_padding=1
                    ), 
                    Branch_conv_conv(           # Branch 3
                        in_channels=96, 
                        hidden_channels=16, 
                        out_channels=16, 
                        conv1_kernel_size=1, 
                        conv1_stride=1, 
                        conv1_padding=0, 
                        conv2_kernel_size=5, 
                        conv2_stride=1, 
                        conv2_padding=2
                    ), 
                    Branch_conv(                # Branch 4
                        in_channels=96, 
                        out_channels=16, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0
                    )
                ]
            ),    # out_channels=112
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception_3 = Inception(
            branches=[
                Branch_maxpool_conv(        # Branch 1
                    in_channels=112, 
                    out_channels=16, 
                    maxpool_kernel_size=3, 
                    maxpool_stride=1, 
                    maxpool_padding=1, 
                    conv_kernel_size=1, 
                    conv_stride=1, 
                    conv_padding=0
                ), 
                Branch_conv_conv(           # Branch 2
                    in_channels=112, 
                    hidden_channels=16, 
                    out_channels=64, 
                    conv1_kernel_size=1, 
                    conv1_stride=1, 
                    conv1_padding=0, 
                    conv2_kernel_size=3, 
                    conv2_stride=1, 
                    conv2_padding=1
                ), 
                Branch_conv_conv(           # Branch 3
                    in_channels=112, 
                    hidden_channels=16, 
                    out_channels=64, 
                    conv1_kernel_size=1, 
                    conv1_stride=1, 
                    conv1_padding=0, 
                    conv2_kernel_size=5, 
                    conv2_stride=1, 
                    conv2_padding=2
                ), 
                Branch_conv(                # Branch 4
                    in_channels=112, 
                    out_channels=16, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0
                )
            ]
        )   # out_channels=160

        self.full_connection = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
            nn.Flatten(), 
            nn.Linear(160*1*1, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.inception_3(x)
        x = self.full_connection(x)
        return x

def train(loader: DataLoader, model: nn.Module, criteria, optimizer, scheduler, epoch):
    model.train()

    total_loss = 0
    total = 0
    correct = 0
    pbar = tqdm(loader, desc=f"EPOCH: {epoch}/{EPOCHS}", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criteria(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predict = torch.max(output.detach(), 1)
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        pbar.set_postfix(
            loss=f"{(total_loss / (i+1)):.4f}", 
            train_acc=f"{(100*correct/total):.2f}%"
        )

    scheduler.step()
    avg_loss = total_loss / len(loader)
    accuracy = 100*correct / total
    print(f"EPOCH: {epoch}/{EPOCHS} | loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%")

def test(loader: DataLoader, model: nn.Module) -> float:
    model.eval()

    total = 0
    correct = 0
    pbar = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            total += labels.size(0)
            _, predict = torch.max(output.detach(), 1)
            correct += (predict == labels).sum().item()
            pbar.set_postfix(
                acc=f"{(100*correct/total):.2f}%"
            )
    accuracy = 100*correct / total
    print(f"Test Acc:{accuracy:.2f}%")
    return accuracy

if __name__ == '__main__':
    print(f"Using device {device}")
    model = GoogLeNet().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=EPOCHS
    )
    criteria = nn.CrossEntropyLoss()
    final_acc = 0

    for epoch in range(1, EPOCHS+1):
        train(
            loader=     train_loader, 
            model=      model, 
            criteria=   criteria, 
            optimizer=  optimizer,
            scheduler=  scheduler,  
            epoch=      epoch
        )
        final_acc = test(test_loader, model)
        print('-'*30)
    print(f"Final Acc: {final_acc:.2f}%")