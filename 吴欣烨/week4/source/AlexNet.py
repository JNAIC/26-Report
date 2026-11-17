import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE      = 128
EPOCHS          = 15
LEARNING_RATE   = 0.001
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

transforms_test = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

train_data = torchvision.datasets.CIFAR10(
    root='./data', 
    download=True, 
    train=True, 
    transform=transform_train
)
test_data = torchvision.datasets.CIFAR10(
    root='./data', 
    download=True, 
    train=False, 
    transform=transforms_test
)

class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolution = nn.Sequential(
            nn.Conv2d(          # Conv1
                in_channels=3, 
                out_channels=64, 
                kernel_size=5, 
                padding=2, 
                stride=2
            ), 
            nn.ReLU(), 
            nn.Conv2d(          # Conv2
                in_channels=64, 
                out_channels=128, 
                kernel_size=3, 
                padding=1
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # TODO: try 3*3, padding=1, stride=2
                                                            #    It may reduce overfitting
            nn.Conv2d(          # Conv3
                in_channels=128, 
                out_channels=256, 
                kernel_size=3, 
                padding=1
            ), 
            nn.ReLU(), 
            nn.Conv2d(          # Conv4
                in_channels=256, 
                out_channels=256, 
                kernel_size=3, 
                padding=1
            ), 
            nn.ReLU(), 
            nn.Conv2d(          # Conv5
                in_channels=256, 
                out_channels=256, 
                kernel_size=3, 
                padding=1
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.full_connection = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(256*4*4, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.full_connection(x)
        return x

def train(loader: DataLoader, model: nn.Module, criteria, optimizer, epoch: int):
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

        total_loss += loss.item()
        _, predict = torch.max(output.detach(), 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        pbar.set_postfix(
            loss=f"{(total_loss / (i+1)):.4f}", 
            acc=f"{(100*correct / total):.2f}%"
        )

    avg_loss = total_loss / len(loader)
    train_accuracy = 100 * correct / total
    print(f"EPOCH: {epoch}/{EPOCHS} | loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}%")


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
            _, predict = torch.max(output.detach(), 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            accuracy = 100*correct / total
            pbar.set_postfix(acc=f"{accuracy:.2f}%")

    test_accuracy = 100 * correct / total
    print(f"Test Acc: {test_accuracy:.2f}%")
    print('-'*30)
    return test_accuracy

if __name__ == '__main__':
    print(f"Using device {device}")
    train_dataloader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    model = AlexNet().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    final_acc = 0

    for epoch in range(1, EPOCHS+1):
        train(
            loader=     train_dataloader, 
            model=      model, 
            criteria=   criteria, 
            optimizer=  optimizer, 
            epoch=      epoch
        )
        final_acc = test(test_dataloader, model)
    print(f"\nFinal Accuracy: {final_acc:.2f}%")