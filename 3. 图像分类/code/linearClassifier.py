import torch
import torch.nn as nn


# 定义线性分类器
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32) # 将3*32*32的矩阵转换为向量
        x = self.softmax(self.fc(x))
        return x
