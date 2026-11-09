import numpy as np
from SimpleCNN import SimpleCNN
from dataset.mnist import load_mnist
from common.optimizer import *
from common.initialization import *

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True, flatten=False)
epoches = 3000
batch_size = 100
train_size = x_train.shape[0]
model = SimpleCNN(100, 10)
optimizer = Adam()

for i in range(1, epoches+1):
    mark = np.random.choice(train_size, batch_size)
    x_batch = x_train[mark]
    y_batch = y_train[mark]
    gradient = model.gradient(x_batch, y_batch)
    optimizer.step(model.params, gradient)
    if i%1000 == 0:
        print(i)
        train_acc = model.accuracy(x_train[:100], y_train[:100])
        test_acc = model.accuracy(x_test[:100], y_test[:100])
        loss = model.loss(x_batch, y_batch)
        print('-'*30)
        print(f"Epoch {i}: \n")
        print(f"accuracy on train set: {train_acc}")
        print(f"accuracy on test set: {test_acc}")
        print(f"loss: {loss}")