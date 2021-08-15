import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from linearClassifier import *
from MLP import *
from CNN import *



net_type = input('请输入要运行的分类器，【1】为softmax线性分类器，【2】为多层感知机，【3】为卷积神经网络')
while net_type not in ('1','2','3'):
    print('输入不合法，请重新输入')
    net_type = input('请输入要运行的分类器，【1】为softmax线性分类器，【2】为多层感知机，【3】为卷积神经网络')


# 读入所有训练集数据
trainset = []
train_images = np.load("cifar10_train_images.npy")
train_labels = np.load("cifar10_train_labels.npy")
for i in range(50000):
    data = torch.Tensor(train_images[i])
    label = train_labels[i]
    trainset.append([data, label])
# 用DataLoader类保存数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0)
\
## 创建CNN实例
if net_type == '1':
    net = Net()
elif net_type == '2':
    net = Net2()
else:
    net = Net3()

y = []
criterion = nn.CrossEntropyLoss()           # 损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001)   # 批梯度下降，学习率
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9,0.99))
t1 = time.time()
for epoch in range(100):        # 学习次数
    running_loss = 0.0          # 损失函数值
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()   # 将参数梯度清零
        outputs = net(inputs)   # 前向传播
        loss = criterion(outputs, labels)   # 计算损失
        loss.backward()         # 反向传播
        optimizer.step()        # 更新梯度
        print(outputs.shape, labels.shape)
        running_loss += loss.item()     # 累加当前损失，用于计算样本平均损失值
        if i+1 == 500:
            y.append(running_loss/100)
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
t2 = time.time()           
print('Finished Training, time used:', t2-t1)
plt.plot([i for i in range(len(y))],y)
plt.show()
# 读入验证集
testset = []
test_images = np.load("cifar10_test_images.npy")
test_labels = np.load("cifar10_test_labels.npy")
for i in range(10000):
    data = torch.Tensor(test_images[i])
    label = test_labels[i]
    testset.append([data, label])
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)
# 验证
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
