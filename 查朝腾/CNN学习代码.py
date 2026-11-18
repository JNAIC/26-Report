import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1. 准备数据
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)


# 2. 选择设备
# torch.cuda.is_available() 只说明"有显卡"，不保证这版 PyTorch 带了
# 适配该显卡的计算内核。所以这里真刀真枪在 GPU 上算一步：
# 算得动才用 GPU，否则退回 CPU。
def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = (torch.ones(1, device="cuda") + 1).sum().item()
        return "cuda" if probe == 2 else "cpu"
    except Exception:
        print("检测到 GPU 但当前 PyTorch 不支持该显卡，自动改用 CPU 训练。")
        return "cpu"


device = pick_device()
print(f"当前使用的设备: {device}")


# 3. 搭建 CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)

# 4. 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. 训练循环（返回这一轮的平均 loss，供后面画曲线用）
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        if batch % 200 == 0:
            print(f"  当前 batch {batch}，loss = {loss.item():.4f}")
    return total_loss / batch_count


epochs = 3
epoch_losses = []  # 记录每轮平均 loss，后面画"loss 下降曲线"
for t in range(epochs):
    print(f"\n第 {t + 1} 轮训练开始")
    avg_loss = train(train_dataloader, model, loss_fn, optimizer)
    epoch_losses.append(avg_loss)
    print(f"第 {t + 1} 轮结束，平均 loss = {avg_loss:.4f}")
print("\n训练完成！")


# 6. 测试准确率（顺便统计混淆矩阵：哪些数字被认成了哪些）
def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)  # 行=真实数字，列=预测数字
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            labels = pred.argmax(1)
            correct += (labels == y).type(torch.float).sum().item()
            total += y.size(0)
            # 把每张图记进 10x10 的格子：真实是 y、预测是 labels 的那一格 +1
            confusion += torch.bincount(y * 10 + labels, minlength=100).reshape(10, 10).cpu()
    acc = correct / total
    print(f"测试集准确率：{acc * 100:.2f}%")
    return acc, confusion


acc, confusion = test(test_dataloader, model)

# =============================================================================
# 7. 把效果画出来（3 张图：loss 曲线 / 预测结果 / 混淆矩阵）
# =============================================================================
# 中文字体设置，否则图里的中文会变成"方块"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --- 图 1：loss 下降曲线（看模型是否在"变聪明"）---
plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), epoch_losses, marker="o", color="royalblue", linewidth=2)
plt.title("训练 loss 曲线（越低越好）")
plt.xlabel("第几轮训练（epoch）")
plt.ylabel("平均 loss")
plt.grid(alpha=0.3)
plt.savefig("效果1_loss曲线.png", dpi=120, bbox_inches="tight")

# --- 图 2：预测结果网格（把测试集图片原样显示 + 标注预测/真实）---
X, y = next(iter(test_dataloader))  # 取测试集第一批
model.eval()
with torch.no_grad():
    pred_labels = model(X[:25].to(device)).argmax(1).cpu()  # 前 25 张的预测结果

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X[i].squeeze(), cmap="gray")  # 显示数据集里的原始图片
    ok = pred_labels[i] == y[i]
    plt.title(
        f"预测:{pred_labels[i]} 真实:{y[i]}",
        color="green" if ok else "red",
        fontsize=10,
    )
    plt.axis("off")
plt.suptitle("测试集预测结果（绿=对，红=错）", fontsize=14)
plt.savefig("效果2_预测结果.png", dpi=120, bbox_inches="tight")

# --- 图 3：混淆矩阵（对角线 = 答对；偏离对角线 = 认错成了别的数字）---
plt.figure(figsize=(7, 6))
plt.imshow(confusion, cmap="Blues")
plt.title("混淆矩阵（对角线越亮越好）")
plt.xlabel("预测是几")
plt.ylabel("真实是几")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        v = confusion[i, j].item()
        if v:
            plt.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > confusion.max().item() / 2 else "black",
                     fontsize=9)
plt.colorbar()
plt.savefig("效果3_混淆矩阵.png", dpi=120, bbox_inches="tight")

print("已生成 3 张图：效果1_loss曲线.png / 效果2_预测结果.png / 效果3_混淆矩阵.png")
plt.show()  # 弹出图片窗口（关掉窗口程序才继续）
