import numpy as np
import random

# 基于距离选择中心点
def genCenterPoints(train_images, k):
    # 第一个中心点随机选择
    center_points = [train_images[random.randint(0,train_images.shape[0]-1)].tolist()]
    while len(center_points) != k:
        print(len(center_points))
        dist = None
        point = None
        # 遍历每个样本点
        for each in train_images:
            # 不考虑已经作为中心点的样本点
            if each.tolist() in center_points:
                continue
            # 计算当前样本到各个中心点的距离之和
            tmp_dist = np.sum(np.linalg.norm(np.asarray(center_points) - each,axis = 1))
            if dist==None or tmp_dist<dist:
                point = each
                dist = tmp_dist
        center_points.append(point.tolist())
    return np.asarray(center_points)
    

# 对样本集进行分类
def classification(center_points,train_images, k):
    # r[i][j]表示第i个样本是(1)否(0)属于第j类
    r = np.zeros((train_images.shape[0],k)).astype(np.int16)
    for i in range(train_images.shape[0]):
        distance = np.linalg.norm(center_points - train_images[i], axis=1)
        r[i][np.argwhere(distance==np.min(distance))[0][0]] = 1
    return r

# 更新中心点位置
def updateCenterPoints(r, train_images, k):
    center_points = np.empty((k,train_images.shape[1]))
    for i in range(k):
        center_points[i] = np.sum(train_images*(np.repeat(r[:,i].reshape(1,-1),\
train_images.shape[1],axis=0).T),axis=0) / np.sum(r[:,i])
    return center_points
    
# 对每个聚类打上标签
def genLabels(r, train_labels, k):
    labels = np.empty(10)
    # 第一层循环遍历每个聚类
    for i in range(k):
        # 第二层循环统计该类中样本属于某一类的个数
        max_count = 0
        max_label = -1
        for j in range(k):
            tmp = np.sum(train_labels[r[:,i]==1]==j)
            if tmp > max_count:
                max_count = tmp
                max_label = j
        labels[i] = max_label
    return labels

    
if __name__ == "__main__":
    train_images = np.load("train-images.npy")
    train_labels = np.load("train-labels.npy")
    test_images = np.load("test-images.npy")
    test_labels = np.load("test-labels.npy")

    # 设置基本参数
    k = 10  # 类别数
    t = 100 # 迭代次数

    # 生成初始中心
    # 完全随机
    center_points = np.random.choice(train_images.shape[0], k, replace = False).tolist()
    for i in range(len(center_points)):
        center_points[i] = train_images[center_points[i]]
    center_points = np.asarray(center_points)
    # 基于距离
    center_points = genCenterPoints(train_images, k)

    # 进行迭代
    for i in range(t):
        print(i)
        # 对训练样本进行分类
        r = classification(center_points,train_images, k)
        # 更新中心点位置
        center_points = updateCenterPoints(r, train_images, k)

    # 对每个聚类打上标签
    labels = (genLabels(r, train_labels, k)).astype(np.int16)

    # 验证结果
    right = 0
    for i in range(test_images.shape[0]):
        distance = np.linalg.norm(center_points - test_images[i], axis=1)
        predict = labels[np.argwhere(distance==np.min(distance))[0][0]]
        if predict == test_labels[i]:
            right+=1
    print(right,right/test_images.shape[0])
    
