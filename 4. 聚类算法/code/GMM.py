import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import random


def genGMM(k,t,train_images):
    # 每个高斯分布的均值，初始化为随机的k个样本点的位置
    train_num = train_images.shape[0]
    gauss_means = np.asarray(train_images[random.sample([i for i in range(train_num)],k)])
    # 每个高斯分布的协方差矩阵
    # 对角且相等
    gauss_var = np.asarray([(1*np.eye(train_images.shape[1])) for j in range(k)])
    # 对角不相等
    # gauss_var = np.asarray([np.diag(np.random.rand(train_images.shape[1])) for j in range(k)])
    # 一般矩阵
    # gauss_var = []
    #for j in range(k):
    #    tmp = np.random.rand(train_images.shape[1],train_images.shape[1])
    #    tmp = np.triu(tmp)
    #    tmp = tmp + tmp.T - 2*np.diag(tmp.diagonal()) + np.diag([1 for i in range(train_images.shape[1])])
    #    gauss_var.append(tmp)
    #gauss_var = np.asarray(gauss_var)
    # gamma[i][j] 表示样本i属于第j个高斯分布的概率
    gamma = np.ones((train_num, k)) / k
    # 每个分布的比重
    pi = gamma.sum(axis=0) / gamma.sum()
    # 开始迭代
    for i in range(t):
        # E-step
        # 计算对数似然函数
        pdf = np.zeros((train_num, k))
        for j in range(k):
            pdf[:, j] = pi[j] * multivariate_normal(gauss_means[j], gauss_var[j],allow_singular=True).pdf(train_images)
        log_likehood = np.mean(np.log(pdf.sum(axis=1)))
        print(log_likehood)
        # M-step
        # 更新各个分布的参数
        gamma = pdf / pdf.sum(axis=1).reshape(-1,1)
        pi = gamma.sum(axis=0) / gamma.sum()
        for j in range(k):
            gauss_means[j] = np.average(train_images, axis=0, weights=gamma[:, j])
            cov = [np.dot((train_images[t]-gauss_means[j]).reshape(-1,1),(train_images[t]-gauss_means[j]).reshape(1,-1)) for t in range(train_num)]
            cov_sum = np.zeros((train_images.shape[1], train_images.shape[1]))
            for l in range(train_num):
                cov_sum += gamma[l][j] * cov[j]
            gauss_var[j] = cov_sum / np.sum(gamma[:,j])
    return [gauss_means, gauss_var, pi]


if __name__ == "__main__":
    # 读入数据
    train_images = np.load("train-images.npy")/255
    train_labels = np.load("train-labels.npy")
    test_images = np.load("test-images.npy")/255
    test_labels = np.load("test-labels.npy")
    train_num = train_images.shape[0]

    
    pca_model = PCA(n_components=50)
    pca_model.fit(train_images.T)
    train_images = pca_model.components_.T
    pca_model.fit(test_images.T)
    test_images = pca_model.components_.T
    
    # 将训练数据按照类别分组
    type_num = 10   # 不同标签的个数
    tmp = [[] for i in range(type_num)]
    for i in range(train_num):
        tmp[train_labels[i]].append(train_images[i])
    # train_images[i][j]表示类型为i的第j个样本
    train_images = tmp[:][:]
    
    # 设置基本参数
    k = 50      # 每个GMM中高斯模型的个数
    t = 10      # 单个GMM迭代次数
    GMM = []    # 记录每个类的GMM
    for i in range(type_num):
        print('-')
        GMM.append(genGMM(k,t,np.asarray(train_images[i])))
    
    # 验证
    right = 0
    for l in range(test_images.shape[0]):
        print(l)
        prob = []
        for i in range(type_num):
            pdf = np.zeros(k)
            for j in range(k):
                pdf[j] = GMM[i][2][j] * multivariate_normal(GMM[i][0][j], GMM[i][1][j],allow_singular=True).pdf(test_images[l])
            prob.append(pdf.sum())
        prob = np.asarray(prob)
        ans = np.argwhere(prob==np.max(prob))[0][0]
        if ans == test_labels[l]:
            right+=1
    print(right, right/test_images.shape[0])
