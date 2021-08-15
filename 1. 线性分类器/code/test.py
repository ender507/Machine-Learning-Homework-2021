import numpy as np
import random

# 传入向量x，返回softmax处理后的向量
def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
    
# 传入标签y，返回相关的独热向量
def onehot(x):
    res = np.zeros((1,label_num))
    res[0][x] = 1
    return res

def valid():
    # 验证
    total = test_images.shape[0]
    right2 = 0
    for i in range(total):
        ans = softmax(np.dot(test_images[i].reshape(1, var_num),W2))
        if np.where(ans == np.max(ans))[1][0] == test_labels[i]:
            right2 += 1  
    print('预测正确个数（均方误差）：',right2)

def validInTrain():
    # 验证
    total = train_images.shape[0]
    right2 = 0
    for i in range(total):
        ans = softmax(np.dot(train_images[i].reshape(1, var_num),W2))
        if np.where(ans == np.max(ans))[1][0] == train_labels[i]:
            right2 += 1  
    print('预测正确个数（均方误差）：',right2)
   
# 读入训练和测试数据
train_images = np.load("train-images.npy")
train_labels = np.load("train-labels.npy")
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")

# 设置基本变量和超参数
train_num = train_images.shape[0]   # 训练集大小
var_num = train_images.shape[1]     # 训练样本特征数
label_num = 10                      # 类别总数
W = np.random.rand(var_num, label_num)# 随机初始化权值矩阵
learn_rate = 0.01                   # 学习率
train_times = 6000                  # 训练次数
mini_batch_size = 60000               # minibatch大小

# 训练
W2 = W[:,:]
for t in range(train_times):
	# 初始化梯度为0  
    grad2 = np.zeros((var_num, label_num)) 
    begin_id = (t % (train_num // mini_batch_size)) * mini_batch_size # 计算batch编号
    # 计算梯度
    for i in range(begin_id, begin_id+ mini_batch_size):
        # 均方误差
        softmax_xW = softmax(np.dot(train_images[i].reshape(1, var_num),W2))
        grad2 = grad2 + 2 * train_images[i].T.reshape(var_num,1)\
               .dot(softmax_xW - onehot(train_labels[i]))\
               #.dot(np.diag(softmax_xW.reshape(10)) - np.dot(softmax_xW,softmax_xW.T))
    grad2 /= mini_batch_size
    # 梯度下降
    W2 = W2 - learn_rate * grad2
    if t%100==0:
        print(t)
        validInTrain()




