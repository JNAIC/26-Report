# =============================================================================
# 【卷积神经网络（CNN）入门完整教程 —— 逐行大白话注释版】
#
# 这个文件做的事一句话概括：
#   教电脑认识手写数字（0~9），训练完还会【画图展示效果】：
#     图1 loss 曲线 —— 看模型是不是在一步步变聪明
#     图2 预测结果 —— 把测试集图片原样显示，标注模型猜了几
#     图3 混淆矩阵 —— 看哪些数字容易被认错
#
# 阅读顺序建议：从上往下读，每个 # 注释都解释了"这行代码在干嘛、为什么需要它"。
# =============================================================================


# -----------------------------------------------------------------------------
# 第 0 步：导入工具包
# -----------------------------------------------------------------------------
# import 的意思就是"把别人的工具拿过来用"。
# torch（PyTorch）是目前最流行的深度学习框架之一，
# 你可以把它理解成"造神经网络的乐高积木套装"。
import torch

# nn = neural network（神经网络）。里面装的全是搭网络用的零件：
# 卷积层、激活函数、池化层、全连接层……后面都会用到。
from torch import nn

# DataLoader 的作用是"发牌员"：把一大堆训练数据切成一小批一小批（batch），
# 每次喂给模型一小批，而不是一口气全塞进去（内存会爆，而且效果也更差）。
from torch.utils.data import DataLoader

# datasets 里内置了经典数据集，比如 MNIST —— 一个包含 6 万张
# 28x28 像素手写数字图片的数据集，是深度学习界的 "Hello World"。
from torchvision import datasets

# ToTensor 是一个转换器：把图片从"像素值 0~255 的普通格式"
# 转成"0.0~1.0 之间的小数张量"。神经网络更喜欢吃这种归一化后的数据。
from torchvision.transforms import ToTensor

# matplotlib 是画图库：折线图、图片展示、热力图都靠它。
# 我们用它把训练成果"可视化"，不然只看终端里的数字太抽象了。
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 第 1 步：准备数据（把饭菜做好端上来）
# -----------------------------------------------------------------------------
# MNIST 数据集分两份：
#   train=True  → 训练集（6 万张）：给模型"上课用的教材"
#   train=False → 测试集（1 万张）：模型学完后"考试用的卷子"
# download=True 表示本地没有就自动联网下载；transform=ToTensor() 表示
# 每读进来一张图，顺手转换成张量格式。
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# DataLoader = 发牌员。
#   batch_size=64：每次发 64 张图给模型（一个"批"）。
#                   批大小是超参数：太小学得慢，太大吃内存。
#   shuffle=True ：每次发牌前把牌洗乱。
#                   目的是防止模型按固定顺序死记硬背，学得更通用。
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)


# -----------------------------------------------------------------------------
# 第 2 步：选择设备（用显卡还是用 CPU 干活）
# -----------------------------------------------------------------------------
# GPU（显卡）擅长同时算几万个小数字，训练神经网络比 CPU 快几十倍。
# 但这里有个坑：
#   torch.cuda.is_available() 只能告诉你"电脑有能用的显卡"，
#   却不能保证你装的这版 PyTorch 带了适配这块显卡的计算内核
#   （显卡太新、或 PyTorch 版本没带对应内核时，它照样返回 True，
#    但一算就崩："no kernel image is available..."）。
#
# 所以正确做法：不听它嘴上说，直接在 GPU 上真算一步试试。
# 算得动 → 用 GPU；算不动 → 退回 CPU（慢点但 100% 能跑）。
def pick_device():
    if not torch.cuda.is_available():
        # 连显卡都没有，直接用 CPU
        return "cpu"
    import warnings
    try:
        with warnings.catch_warnings():
            # 静音那堆"你的 GPU 太新"的警告，让输出干净点
            warnings.simplefilter("ignore")
            # 关键一行：在 GPU 上真的算一次 1+1。
            # 如果内核缺失，错误会在这一步立刻抛出来，被下面的 except 接住。
            # .item() 会强制等待计算完成（同步），确保错误不是"迟到"才报。
            probe = (torch.ones(1, device="cuda") + 1).sum().item()
        # 能算到结果 2，说明 GPU 没问题
        return "cuda" if probe == 2 else "cpu"
    except Exception:
        # GPU 试算失败 → 好消息：程序不会崩，改成用 CPU 继续训练
        print("检测到 GPU 但当前 PyTorch 不支持该显卡，自动改用 CPU 训练。")
        return "cpu"


device = pick_device()
print(f"当前使用的设备: {device}")  # f-string：把花括号里的变量值塞进字符串里打印


# -----------------------------------------------------------------------------
# 第 3 步：搭建 CNN（设计神经网络的"流水线"）
# -----------------------------------------------------------------------------
# nn.Module 是所有神经网络的"基类"，相当于毛坯房，
# 我们继承它，再往里面装修（添加层、定义数据怎么流动）。
class CNN(nn.Module):

    def __init__(self):
        # super().__init__() 是固定写法：先让 PyTorch 官方的初始化逻辑跑一遍，
        # 否则后面 self.xxx = ... 会报错。记成"开工前先跟物业报备"即可。
        super().__init__()

        # ============ 卷积部分（负责"看图找特征"）============
        # Sequential 的意思是"按顺序排队"，数据会从上到下依次流过每一层。
        self.conv_stack = nn.Sequential(

            # 【卷积层 Conv2d】= CNN 的灵魂。
            # 可以想象成一个"小放大镜"在图片上滑动，
            # 每到一处就做一次加权求和，提取局部特征（比如横线、圆弧）。
            #   1  → 输入通道数：图片是灰度图，只有 1 层颜色（彩色图则是 3 层 RGB）
            #   16 → 输出通道数：用 16 个不同的"放大镜"去提取 16 种不同的特征
            #   kernel_size=3 → 放大镜大小是 3x3 像素
            #   padding=1     → 图片四周补 1 圈 0，让输出尺寸不缩水
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输出: 16 个 28x28 特征图

            # 【ReLU 激活函数】= 给模型引入"非线性"。
            # 它极其简单：负数全变 0，正数不变。
            # 为什么需要？如果没有它，一百层网络叠起来也等价于一层线性变换，
            # 只能画直线分界，根本认不出复杂的数字形状。
            # 好比人脑：信号不够强就不反应（阈值），这样才有"判断"。
            nn.ReLU(),

            # 【最大池化 MaxPool2d(2)】= 压缩降采样。
            # 每 2x2 的小格子里只保留最大的那个数，其余丢掉。
            # 好处：① 数据量减半，算得更快；② 只保留"最明显的特征"，
            #         对图片轻微的平移、缩放不再敏感（更鲁棒）。
            nn.MaxPool2d(2),   # 28x28 → 14x14

            # 第二层卷积：把 16 种初级特征再组合成 32 种更高级的特征。
            # 如果第一层认的是"横线竖线"，这一层认的可能就是"圆圈、交叉"。
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 输出: 32 个 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14x14 → 7x7

            # 走到这里，一张 28x28 的图片变成了 32 张 7x7 的"特征图"。
        )

        # ============ 分类部分（负责"下结论"）============
        self.classifier = nn.Sequential(

            # 【Flatten 展平】把 32x7x7 的立方体压成一条 1568 个数的长队列。
            # 因为后面的全连接层只吃"一维数组"，不吃二维图片。
            nn.Flatten(),                                  # 32*7*7 = 1568

            # 【全连接层 Linear】= 最经典的神经元连接方式：
            # 输入的每个数都和输出的每个数连一根线，每根线上有个"权重"。
            # 1568 个输入 → 128 个神经元：让网络在这里"综合推理"。
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            # 最后一层输出 10 个分数 —— 对应 0~9 这 10 个数字。
            # 注意这里【不加】ReLU！因为要的是原始分数（logit），
            # 分数最高的那个类别就是模型的预测结果。
            nn.Linear(128, 10),
        )

    # forward 定义"数据从进到出的路线"，PyTorch 会在调用 model(x) 时自动执行。
    # 你可以理解成工厂的生产线：卷积流水线看图 → 分类流水线下结论。
    def forward(self, x):
        x = self.conv_stack(x)     # 先看图提特征
        x = self.classifier(x)     # 再把特征翻译成 10 个类别的分数
        return x                   # 返回的 shape 是 [批大小, 10]


# 实例化（造出一台真正的网络机器），并搬到选定的设备上（.to(device)）。
# model.parameters() 收集了网络里所有待学习的"旋钮"（权重和偏置）。
model = CNN().to(device)


# -----------------------------------------------------------------------------
# 第 4 步：损失函数和优化器（出考题 + 定改进方向）
# -----------------------------------------------------------------------------
# 【损失函数 loss】= 衡量"模型现在有多菜"。
# CrossEntropyLoss 是分类任务的标配：它比较模型输出的 10 个分数
# 和正确答案，差距越大 loss 越大。训练的目标就是把 loss 搞小。
loss_fn = nn.CrossEntropyLoss()

# 【优化器 optimizer】= 负责根据 loss 去调整那一大堆权重。
# Adam 是目前最常用的优化器，特点是自带"学习率自适应"，
# 对新手非常友好，基本不用怎么调就能收敛。
# lr=0.001（学习率）= 每次调整的步子大小：
#   步子太大 → 跨过最低点来回震荡；
#   步子太小 → 学得慢如蜗牛。0.001 是个经典起步值。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------------------------------------------------------
# 第 5 步：训练循环（真正"上课"的过程）
# -----------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()  # 切换到"训练模式"（会启用 Dropout、BatchNorm 等训练专用行为）

    total_loss = 0.0   # 累加这一轮所有 batch 的 loss
    batch_count = 0    # 一共过了多少个 batch

    # enumerate(dataloader)：一边发牌一边报数，batch 从 0 开始编号。
    for batch, (X, y) in enumerate(dataloader):
        # X = 这一批的图片，y = 这一批图片对应的正确答案（数字几）
        # .to(device)：把数据搬到 GPU/CPU 上，数据和模型必须在同一个设备上。
        X, y = X.to(device), y.to(device)

        # ① 前向传播：把图片喂进网络，算出预测结果（10 个分数）
        pred = model(X)
        # ② 算损失：预测和正确答案差多少 → 得到一个 loss 数值
        loss = loss_fn(pred, y)

        # ③ 清空梯度。PyTorch 默认会把每次算的梯度累加起来，
        #    不清零的话，这一步的梯度会混进下一步 → 越算越错。
        #    这是新手最常踩的坑之一，务必记得写在 backward() 之前。
        optimizer.zero_grad()
        # ④ 反向传播：沿着网络从后往前，算出每个权重对 loss 的"责任"（梯度）。
        #    这一步是深度学习的核心数学 —— 链式法则的自动求导。
        loss.backward()
        # ⑤ 更新权重：优化器按梯度方向，把所有旋钮拧一点点，
        #    让 loss 下一步能变小一点。这就叫"学习"。
        optimizer.step()

        # 把 loss 记进小账本，最后算平均值用来画曲线
        total_loss += loss.item()
        batch_count += 1

        # 每 200 个 batch 打印一次进度，免得你盯着黑屏以为死机了。
        # loss.item() 把张量取成普通数字；:.4f 保留 4 位小数。
        if batch % 200 == 0:
            print(f"  当前 batch {batch}，loss = {loss.item():.4f}")

    # 返回这一轮的平均 loss —— 后面画"loss 下降曲线"就靠它
    return total_loss / batch_count


# epoch（轮）= 把整个训练集完整学一遍。
# 一遍不够，就多来几遍，通常学得越多成绩越好（但太多会"背题"= 过拟合）。
epochs = 3
epoch_losses = []  # 空列表：专门存每一轮的平均 loss，画曲线的数据源
for t in range(epochs):
    print(f"\n第 {t + 1} 轮训练开始")
    avg_loss = train(train_dataloader, model, loss_fn, optimizer)
    epoch_losses.append(avg_loss)  # append = 往列表末尾追加一个数
    print(f"第 {t + 1} 轮结束，平均 loss = {avg_loss:.4f}")
print("\n训练完成！")


# -----------------------------------------------------------------------------
# 第 6 步：测试准确率（期末考试）+ 统计"错题分布"
# -----------------------------------------------------------------------------
def test(dataloader, model):
    # 切换到"评估模式"。考试时不能作弊，也不能瞎调整。
    model.eval()
    correct = 0  # 答对了几题
    total = 0    # 一共几题

    # 【混淆矩阵】一个 10x10 的格子：
    #   行 = 真实是几，列 = 被预测成几。
    #   比如第 4 行第 9 列的数字 = "真实是 4 却被认成 9" 的次数。
    #   训练好之后对角线应该很亮（因为大部分该答对），其他格子越暗越好。
    confusion = torch.zeros(10, 10, dtype=torch.long)

    # torch.no_grad()：告诉 PyTorch "这是考试，不用求梯度"。
    # 三个好处：省内存、跑得快、防止误改权重。
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 预测：每张图得到 10 个分数

            # argmax(1)：取 10 个分数里最大值的下标 = 模型认为最像的数字。
            labels = pred.argmax(1)

            # == y：和正确答案比对，对就是 True（当 1），错就是 False（当 0）。
            # .type(torch.float).sum().item()：数一数总共对了几个。
            correct += (labels == y).type(torch.float).sum().item()
            total += y.size(0)  # y.size(0) = 这一批有多少张图

            # 记账进混淆矩阵：
            # 把"真实 y、预测 labels"编码成 y*10+labels 这个数（比如真实4预测9 → 49），
            # bincount 数出 0~99 每个数出现几次，再 reshape 回 10x10 格子。
            # 最后 .cpu() 很关键：算出来的结果在 GPU 上，而 confusion 矢量
            # 建在 CPU 上，两边设备不同会报错，所以先搬回 CPU 再累加。
            confusion += torch.bincount(y * 10 + labels, minlength=100).reshape(10, 10).cpu()

    acc = correct / total
    print(f"测试集准确率：{acc * 100:.2f}%")
    # 这个数字才是衡量模型好坏的标准（用的是模型没见过的测试集）。
    # 一般 MNIST 上 3 轮能达到 97%~99%。
    return acc, confusion  # 把成绩和错题分布一起交出去，供画图使用


acc, confusion = test(test_dataloader, model)


# =============================================================================
# 第 7 步：把效果画出来！（3 张图，这才是"看得见的效果"）
# =============================================================================
# 【中文字体设置】matplotlib 默认字体不带中文，不设置的话
# 图里的中文会显示成一堆"方块"（豆腐块）。Windows 自带微软雅黑，用它。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
# 负号显示设置：不加这行，坐标轴上的负号可能变成方块
plt.rcParams["axes.unicode_minus"] = False

# ---------- 图 1：loss 下降曲线 ----------
# 把 epoch_losses（每轮平均 loss）画成折线：
# 一条持续下降的曲线 = 模型在一轮轮变聪明。
# 如果曲线平了不降 → 说明训练到位了；如果反弹上升 → 出问题了（过拟合等）。
plt.figure(figsize=(7, 4))                       # 新建一张图，尺寸 7x4 英寸
plt.plot(
    range(1, epochs + 1),                        # 横坐标：第 1、2、3 轮
    epoch_losses,                                # 纵坐标：对应的平均 loss
    marker="o",                                  # 每个数据点画个小圆点
    color="royalblue", linewidth=2,              # 线条颜色和粗细
)
plt.title("训练 loss 曲线（越低越好）")           # 图标题
plt.xlabel("第几轮训练（epoch）")                 # 横轴说明
plt.ylabel("平均 loss")                          # 纵轴说明
plt.grid(alpha=0.3)                              # 淡淡的网格线，方便读数
# dpi=120 让图片清晰一点；bbox_inches="tight" 自动裁掉多余白边
plt.savefig("效果1_loss曲线.png", dpi=120, bbox_inches="tight")

# ---------- 图 2：预测结果网格 ----------
# 从测试集里取第一批（64 张），用前 25 张来展示。
X, y = next(iter(test_dataloader))
model.eval()                                     # 展示前也切到评估模式
with torch.no_grad():                            # 同样不开梯度，省资源
    # 把 25 张图喂给模型 → 每张得到 10 个分数 → argmax 取分数最高的当预测
    # .cpu() 把结果从 GPU 搬回 CPU（因为画图只认 CPU 数据）
    pred_labels = model(X[:25].to(device)).argmax(1).cpu()

plt.figure(figsize=(10, 10))                     # 10x10 英寸的正方形大图
for i in range(25):                              # 循环画 25 张小图
    plt.subplot(5, 5, i + 1)                     # 分成 5 行 5 列，这是第 i+1 格
    plt.imshow(X[i].squeeze(), cmap="gray")      # 显示数据集里的原始图片（灰度）
    # squeeze()：把 [1,28,28] 这种带多余维度的张量压成 [28,28]，imshow 才认
    ok = pred_labels[i] == y[i]                  # 预测是否正确
    plt.title(
        f"预测:{pred_labels[i]} 真实:{y[i]}",     # 标题写上"猜的/实际的"
        color="green" if ok else "red",          # 对了绿字，错了红字
        fontsize=10,
    )
    plt.axis("off")                              # 关掉每格的坐标轴，只留图片
plt.suptitle("测试集预测结果（绿=对，红=错）", fontsize=14)  # 大标题
plt.savefig("效果2_预测结果.png", dpi=120, bbox_inches="tight")

# ---------- 图 3：混淆矩阵热力图 ----------
# 把 10x10 的 confusion 画成热力图：颜色越深 = 次数越多。
# 理想情况：只有对角线（0-0, 1-1, ..., 9-9）是深色，
# 其他格子接近白色 —— 那说明模型没把数字认错。
plt.figure(figsize=(7, 6))
plt.imshow(confusion, cmap="Blues")              # 用蓝色深浅表示数值大小
plt.title("混淆矩阵（对角线越亮越好）")
plt.xlabel("预测是几")                           # 横轴 = 模型猜的答案
plt.ylabel("真实是几")                           # 纵轴 = 标准答案
plt.xticks(range(10))                            # 横轴刻度 0~9
plt.yticks(range(10))                            # 纵轴刻度 0~9
# 在每个格子里写上具体次数（0 的格子不写，免得密密麻麻）
for i in range(10):
    for j in range(10):
        v = confusion[i, j].item()
        if v:
            plt.text(
                j, i, str(v),                    # 位置 (列, 行) 和文字内容
                ha="center", va="center",        # 文字居中
                color="white" if v > confusion.max().item() / 2 else "black",
                fontsize=9,                      # 深色底用白字，浅色底用黑字
            )
plt.colorbar()                                  # 右侧色条：颜色对应多少次
plt.savefig("效果3_混淆矩阵.png", dpi=120, bbox_inches="tight")

# 最后：把三张图的文件名告诉用户，然后弹出图片窗口。
# plt.show() 会一直阻塞到你关掉所有图片窗口，程序才结束 —— 方便慢慢看。
print("已生成 3 张图：效果1_loss曲线.png / 效果2_预测结果.png / 效果3_混淆矩阵.png")
plt.show()

# =============================================================================
# 【读完这篇，你已经掌握了 CNN 的完整知识框架，下面是要点回顾】
#
# 1. 完整流程 = 准备数据 → 搭网络 → 定损失和优化器 → 训练 → 测试 → 可视化。
# 2. CNN 的两大分工：
#      卷积+池化 = 看图提特征；全连接 = 根据特征下结论。
# 3. 训练四部曲（背下来）：
#      前向传播算预测 → 算 loss → backward() 求梯度 → step() 改权重。
#    记得每轮先 zero_grad() 清梯度！
# 4. 超参数 = 人手动定的数：batch_size、epochs、lr、卷积核个数等。
#    调这些数就是在"调教"模型。
# 5. 环境坑：torch.cuda.is_available() 为 True 不代表 GPU 真能用，
#    要真算一步才知道 —— 本文件的 pick_device() 就是干这个的。
# 6. 训练完一定要"看见"成果：
#      loss 曲线降不降 → 训练有没有效果；
#      预测网格红不红 → 模型错在哪几张；
#      混淆矩阵斜不亮 → 哪两个数字容易搞混（如 4/9、3/8）。
# =============================================================================
