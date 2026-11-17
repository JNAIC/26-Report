import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from num_recog_model import *

num_train_dataset = torchvision.datasets.MNIST("../Num_Recognize",train=True,transform=torchvision.transforms.ToTensor(),download=True)
num_test_dataset = torchvision.datasets.MNIST("../Num_Recognize",train=False,transform=torchvision.transforms.ToTensor(),download=True)


print(f"训练集的长度为：{len(num_train_dataset)}")
print(f"测试集的长度为：{len(num_test_dataset)}")

train_loader = DataLoader(num_train_dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
test_loader = DataLoader(num_test_dataset,batch_size=32,shuffle=False,num_workers=0,drop_last=False)

model = MyModule()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters,lr=learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 20
for i in range(epoch):
    print(f"————第{i+1}轮训练开始————")

    for data in train_loader:
        imgs,targets = data
        outputs_train = model(imgs)
        loss_train = loss_fn(outputs_train,targets)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 5 == 0:
            print(f"第{total_train_step}轮训练，loss:{loss_train}")

total_test_loss = 0
with torch.no_gard():
    for data in test_loader:
        imgs,targets = data
        outputs_test = model(imgs)
        loss_test = loss_fn(outputs_test,targets)

        total_test_loss += loss_test

print(f"测试集上的整体loss为:{total_test_loss}")