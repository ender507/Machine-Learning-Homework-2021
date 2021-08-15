import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NetDef import *
import random

def trainNet(net, sub_trainloader, zero_labels):
    criterion = nn.CrossEntropyLoss()           # 损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001)   # 批梯度下降，学习率
    for each_trainloader in sub_trainloader:
        for epoch in range(100):        # 学习次数
            running_loss = 0.0          # 损失函数值
            for i, data in enumerate(each_trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()   # 将参数梯度清零
                outputs = net(inputs)   # 前向传播
                for i in range(labels.shape[0]):
                    if labels[i] in zero_labels:
                        labels[i] = 0
                    else:
                        labels[i] = 1
                loss = criterion(outputs, labels)   # 计算损失
                loss.backward()         # 反向传播
                optimizer.step()        # 更新梯度
    print('fin')
    return net


if __name__ == '__main__':

    load_model = False  # 是否读入训练好的模型
    
    # 读入所有训练集数据
    train_images = np.load("cifar10_train_images.npy")
    train_labels = np.load("cifar10_train_labels.npy")
    trainloader = [[] for i in range(10)]    # trainloader列表分别存放10个不同类的样本的data
    for i in range(train_images.shape[0]):
        data = torch.Tensor(train_images[i]).reshape((3,32,32))
        label = train_labels[i]
        trainloader[label].append([data, label])
    # 用DataLoader类保存数据
    for i in range(10):
        trainloader[i] =  torch.utils.data.DataLoader(trainloader[i], batch_size=1000, shuffle=False, num_workers=0)

    # 创建CNN实例
    net_ = Net()#是否是动物
    net0_ = Net()#是否在地上跑
    net00_ = Net()#是否是飞机
    net01_ =Net()#是否是汽车
    net1_ = Net()#是否是哺乳动物
    net10_ = Net()#是否是鸟
    net11_ = Net()#是否可以做家养宠物
    net110_ = Net()#是否是鹿
    net111_ = Net()#是否是猫
    # 十个类分别为：0飞机、1汽车、2鸟、3猫、4鹿、5狗、6青蛙、7马、8船、9卡车
    encoded_label = ["001","011","101","1111","1101","1110","100","1100","000","010"]

    if load_model:
        net_.load_state_dict(torch.load('net_.pkl'))
        net0_.load_state_dict(torch.load('net0_.pkl'))
        net00_.load_state_dict(torch.load('net00_.pkl'))
        net1_.load_state_dict(torch.load('net1_.pkl'))
        net10_.load_state_dict(torch.load('net10_.pkl'))
        net11_.load_state_dict(torch.load('net11_.pkl'))
        net110_.load_state_dict(torch.load('net110_.pkl'))
        net111_.load_state_dict(torch.load('net111_.pkl'))
    else:
        # 训练分类器
        net_ = trainNet(net_, trainloader, [0,1,8,9])
        net0_ = trainNet(net0_, [trainloader[0],trainloader[1],trainloader[8],trainloader[9]],[0,8])
        net00_ = trainNet(net00_, [trainloader[0],trainloader[8]],[8])
        net1_ = trainNet(net1_, [trainloader[2],trainloader[3],trainloader[4],trainloader[5],
                                     trainloader[6],trainloader[7]],[2,6])
        net10_ = trainNet(net10_, [trainloader[2],trainloader[6]],[6])
        net11_ = trainNet(net11_, [trainloader[3],trainloader[4],trainloader[5],trainloader[7]],[4,7])
        net110_ = trainNet(net110_, [trainloader[4],trainloader[7]],[7])
        net111_ = trainNet(net111_, [trainloader[3],trainloader[5]],[5])
        torch.save(net_.state_dict(),'net_.pkl')
        torch.save(net0_.state_dict(),'net0_.pkl')
        torch.save(net00_.state_dict(),'net00_.pkl')
        torch.save(net1_.state_dict(),'net1_.pkl')
        torch.save(net10_.state_dict(),'net10_.pkl')
        torch.save(net11_.state_dict(),'net110_.pkl')
        torch.save(net111_.state_dict(),'net111_.pkl')
        torch.save(net110_.state_dict(),'net110_.pkl')

    # 读入测试集，随机选择一条数据
    test_images = np.load("cifar10_test_images.npy")
    sample = torch.Tensor(test_images[random.randint(0,test_images.shape[0]-1)].reshape((1,3,32,32)))
    net = net_
    # 将输入样例进行编码
    code = ""
    while True:
        outputs = net(sample)
        _, predicted = torch.max(outputs.data, 1)
        code = code + str(np.array(predicted)[0])
        if int(np.array(predicted)[0]) == 0:
            if net == net_:
                net = net0_
            elif net == net0_:
                net = net00_
            elif net == net1_:
                net = net10_
            elif net == net11_:
                net = net110_
            else:
                break
        else:
            if net == net_:
                net = net1_
            elif net == net1_:
                net = net11_
            elif net == net11_:
                net = net111_
            else:
                break
    print('该样本的编码为',code)
    length = len(code)
    sim = input('请输入相似度，范围为0到'+str(length)+':')
    sim = int(sim)
    print('匹配到的相似编码为：')
    for each in encoded_label:
        if len(each)>=sim and each[:sim] == code[:sim]:
                print(each)
        
        
        
