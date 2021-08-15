import numpy as np
import random


def valid():
    # 验证
    total = test_images.shape[0]
    right = 0
    for i in range(total):
        if np.dot(test_images[i],w.T) >= 0:
            ans = 1
        else:
            ans = -1
        if ans == test_labels[i]:
            right += 1  
    print('验证集大小：',total)
    print('预测正确个数（hinge loss）：',right)
    
# 读入训练和测试数据
train_images = np.load("train-images.npy")
train_labels = np.load("train-labels.npy")
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")
# 把标签改为1和-1而不是1和0
train_labels[train_labels==0] = -1
test_labels[test_labels==0] = -1
# 为训练和测试数据增加一维1
tmp = np.zeros((train_images.shape[0],train_images.shape[1]+1))
for i in range(train_images.shape[0]):
    tmp[i] = np.append(train_images[i]/255, 1)
train_images = tmp[:][:]
tmp = np.zeros((test_images.shape[0],test_images.shape[1]+1))
for i in range(test_images.shape[0]):
    tmp[i] = np.append(test_images[i]/255, 1)
test_images = tmp[:][:]

# 设置基本变量和超参数
train_num = train_images.shape[0]   # 训练集大小
var_num = train_images.shape[1]     # 训练样本特征数
w = np.random.rand(var_num)         # 随机初始化权值向量
learn_rate = 0.001                  # 学习率
train_times = 1000                  # 训练次数

# 训练
for t in range(train_times):
    # 初始化梯度为0 
    grad = np.zeros((var_num))
    # 计算梯度
    for i in range(train_num):
        if 1 - np.dot(train_images[i],w) * train_labels[i] < 0:          
            continue
        grad = grad - (np.dot(train_images[i],w) * train_labels[i]) * train_images[i]
    grad /= train_num
    # 梯度下降
    w = w - learn_rate * grad
    if t%100==0:
        print(t)
        # print(np.max(w), np.min(w))
        valid()




