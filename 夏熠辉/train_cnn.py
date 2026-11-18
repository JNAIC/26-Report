import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms

full_train_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

train_data,val_data = random_split(
    full_train_data,
    [50000,10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),
)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


print("训练设备", device, flush=True)
print("学习率", optimizer.param_groups[0]["lr"], flush=True)

train_losses = []
val_accuracies = []

for epoch in range(1, 6):
    train_loss = train_one_epoch(
        model, train_loader, loss_fn, optimizer, device
    )
    val_accuracy = evaluate_accuracy(model, val_loader, device)

    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"第 {epoch} 轮 | "
        f"平均训练损失 {train_loss:.4f} | "
        f"验证准确率 {val_accuracy:.2%}",
        flush=True,
    )


test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=transforms.ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=64)

test_accuracy = evaluate_accuracy(model, test_loader, device)
print(f"测试样本数 {len(test_data)} | 测试准确率 {test_accuracy:.2%}")

torch.save(model.state_dict(), "mnist_cnn.pt")
print("已保存模型权重 mnist_cnn.pt")
