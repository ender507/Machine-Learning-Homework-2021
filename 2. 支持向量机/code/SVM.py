import numpy as np
from sklearn import svm

# 读入数据集
train_images = np.load("train-images.npy")
train_labels = np.load("train-labels.npy")
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")
print("验证集大小：",test_images.shape[0])

# 线性核函数SVC模型
model = svm.SVC(kernel='linear') 
model.fit(train_images, train_labels)# 训练
predicted= model.predict(test_images)# 预测
# 统计
right = 0
for i in range(test_images.shape[0]):
    if predicted[i] == test_labels[i]:
        right += 1
print("分类正确个数（线性核）：",right)
print("准确率（线性核）：",right/test_images.shape[0])


# 高斯核函数SVC模型
model = svm.SVC(kernel='rbf') 
model.fit(train_images, train_labels)
predicted= model.predict(test_images)
right = 0
for i in range(test_images.shape[0]):
    if predicted[i] == test_labels[i]:
        right += 1
print("分类正确个数（高斯核）：",right)
print("准确率（高斯核）：",right/test_images.shape[0])
