import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#   initialize data
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform
)

#   set dataloader for mini-batch training
train_dataloader = DataLoader(
    train_dataset, 
    BATCH_SIZE, 
    shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, 
    BATCH_SIZE, 
    shuffle=True
)

class SimpleCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels =    1,     # grey images only have one channel
                out_channels=   32,     # the number of kernel, also the next conv layer's in_channel
                kernel_size =    3,
                padding     =    1      # to make the matrix shape remain the same
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.affine = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(
                64*7*7,     # 64 is the ouput channel of thelast conv layer
                128,        # the number of neurons, but we call it "channel" in pytorch
            ), 
            nn.ReLU(), 
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.affine(x)
        return x


def train(model, loader, criterion, optimizer, epoch):
    model.train()

    total_loss = 0
    total = 0
    correct = 0

    for image, label in loader:
        image.to(device)
        label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predict==label).sum().item()

    accuracy = 100 * correct / total

    print('-'*30)
    print(f"Epoch: {epoch}/{EPOCHS}")
    print(f"Loss: {total_loss / len(loader)}")
    print(f"Train Accuracy: {accuracy:.2f}%")

def test(model, loader):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for image, label in loader:
            image.to(device)
            label.to(device)

            output = model(image)
            _, predict = torch.max(output, 1)
            total += label.size(0)
            correct += (predict==label).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

if __name__ == '__main__':    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    final_acc = 0

    for epoch in range(1, EPOCHS+1):
        train(model, train_dataloader, criterion, optimizer, epoch)
        final_acc = test(model, test_dataloader)

    print(f"Final Accuracy: {final_acc:.2f}%")
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved successfully")