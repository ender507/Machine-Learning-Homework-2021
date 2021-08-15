# 机器学习与数据挖掘

# Assignment 1实验报告

>姓名：507
>
>学号：18340057
>
>班级：计算机科学二班

## 一、 理论知识

### 1.1 从线性回归到线性多分类

回归是基于给定的特征，对感兴趣的变量进行值的预测的过程。在数学上，回归的目的是建立从输入数值到监督数值的函数：
$$
\hat y=f(x_1,...,x_m)
$$
线性回归限制函数为线性形式，即为：
$$
f(x_1,...x_m)=w_0+w_1x_1+...+w_mx_m=\bold x\bold w
$$
其中，
$$
\bold x = [1,x_1,x_2,...,x_m]\\
\bold w = [w_0,w_1,w_2,...,w_m]^T
$$
也就是找一组参数$\{w_k\}^m_{k=1}$，使得在训练集上，函数与预测值尽可能接近。

对于本次的分类问题来说，线性回归的输出值与分类任务中的目标值不兼容。线性回归的结果范围为全体实数，而对于本次实验的多分类问题，变量结果即属于的类别，换言之，我们期望的结果标签的种类数量和训练样本的总类别数量一致。因此考虑使用softmax函数来将回归结果映射到种类上，从而表示分类结果。对于K分类问题，有：
$$
softmax_i(\bold z)=\frac{e^{z_i}}{\sum^K_{k=1}e^{z_k}}\\
f_i(\bold x)=softmax_i(\bold{xW})=\frac{e^{\bold{xw_i}}}{\sum^K_{k=1}e^{\bold{xw_k}}}
$$
其中，$\bold W$为：
$$
\bold W\triangleq \left[\begin{matrix}{\bold w_1,\bold w_2...,\bold w_K}\end{matrix}\right]
$$
易见，所有类的softmax函数值之和为1。每一类的函数值就为它的概率。

### 1.2 损失函数表示与优化

经过上面的讨论与操作，对于多分类问题，预测结果是在每一类上的概率，即维度数等于类数的向量。与之对应的实际结果可以用独热向量表示，即是本类的那一维度为1，其他维度为0的向量。为了使得预测结果与实际结果尽量接近，我们考虑用损失函数用于衡量预测结果和实际结果的差距。在数学上，该分类问题等价于找到合适的向量$\bold w$，使得损失函数最小化。依据本次实验的要求，损失函数需要分别考虑交叉熵损失和均方误差损失，即损失函数分别为：
$$
L_1(\bold w_1,\bold w_2,...,\bold w_K)=-\frac1N\sum^N_{l=1}\sum^K_{k=1}y_k^{(l)}\log softmax_k(\bold x^{(l)}\bold W)\\
L_2(\bold w_1,\bold w_2,...,\bold w_K)=\frac1N\sum^N_{l=1}\sum^K_{k=1}(softmax_k(\bold x^{(l)}\bold W)-y^{(l)}_k)^2
$$
其中，$y_k^{(l)}$是第$k$个$y^{(l)}$的元素。

考虑使用梯度下降法使得损失函数最小化。两个损失函数的梯度分别为：
$$
\frac{\part L(\bold W)}{\part\bold W}=\frac1N\sum^N_{l=1}\bold x^{(l)T}(softmax(\bold x^{(l)}\bold W)-\bold y^{(l)})\\
\frac{\part L(\bold W)}{\part\bold W}=\frac2N\sum^N_{l=1}\bold x^{(l)T}(softmax(\bold x^{(l)}\bold W)-\bold y^{(l)})*(diag(softmax(\bold x^{(l)}\bold W)-softmax(\bold x^{(l)}\bold W)*softmax(\bold x^{(l)}\bold W)^T)
$$

梯度下降法的参数更新方式为：
$$
\bold W^{(t+1)}=\bold W^{(t)}-r\left.\frac{\part L(\bold W)}{\part\bold W}\right|_{\bold W=\bold W^{(t)}}
$$

其中$r$为学习率。对于凹函数，通过适当的学习率，对模型参数进行迭代更新，最终可以收敛到最小值点。

-----------

## 二、 训练过程

### 2.1 基本参数和功能的设置

首先读入训练和测试数据：

```python
train_images = np.load("train-images.npy")
train_labels = np.load("train-labels.npy")
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")
```

接着设置基本的变量和超参数：

```python
train_num = train_images.shape[0]   # 训练集大小
var_num = train_images.shape[1]     # 训练样本特征数
label_num = 10                      # 类别总数
W = np.random.rand(var_num, label_num)# 随机初始化权值矩阵
learn_rate = 0.01                   # 学习率
train_times = 6000                  # 训练次数
mini_batch_size = 100               # minibatch大小
```

关于超参数的选择，我尝试的不同的超参数，并得到了不同的实验结果，将在之后讨论。

在多分类问题中，还需要实现softmax函数。按照本来softmax函数的定义，给定结果的向量，经softmax函数变换后向量的每个位置应该是当前位置的exp函数值除以全部位置的exp函数值之和。但在实际训练的时候，特征值最大的为三位十进制数。计算自然常数e的几百次方容易导致溢出。我们注意到softmax函数有如下性质：
$$
softmax_i(\bold z)=\frac{e^{z_i}}{\sum^K_{k=1}e^{z_k}}=\frac{ce^{z_i}}{c\sum^K_{k=1}e^{z_k}}=\frac{e^{z_i+\ln c}}{\sum^K_{k=1}e^{z_k+\ln c}}
$$


换言之，如果将输入的向量的每个位置同时加上或者减去一个值，这个向量的softmax结果是不会变的。因此，可以将该向量减去该向量中的最大值，可以将所有的数值变为非正数，经过自然常数e的幂变换后映射到$(0,1]$，从而不会溢出，同时也能求出正确的softmax值：

```python
# 传入向量x，返回softmax处理后的向量
def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

同时，在该问题中，得到的结果标签是一个整数，需要将它转换为独热向量：

```python
# 传入标签y，返回相关的独热向量
def onehot(x):
    res = np.zeros((1,label_num))
    res[0][x] = 1
    return res
```

### 2.2 训练过程

训练集大小为60000，较大，因此我尝试使用了mini-batch的方法。若训练集大小为`train_num`，mini-batch大小为`mini_batch_size`，则一共有`train_num // mini_batch_size`个batch（尽管这个除法的结果一定为整数，但实际python代码中使用整形除法`//`而不使用`/`可以避免之后的数据类型转换）。当训练次数为`t`时，训练的是第`t % (train_num // mini_batch_size)`个batch。将这个数乘以`mini_batch_size`就得到了需要训练的batch的第一个训练样本的id。

代入之前在理论部分讨论过的训练公式和梯度下降公式，就能实现训练过程：

```python
# 训练
W1 = W[:,:]
W2 = W[:,:]
for t in range(train_times):
	# 初始化梯度为0
    grad1 = np.zeros((var_num, label_num))   
    grad2 = np.zeros((var_num, label_num)) 
    begin_id = (t % (train_num // mini_batch_size)) * mini_batch_size # 计算batch编号
    # 计算梯度
    for i in range(begin_id, begin_id+ mini_batch_size):
        # 交叉熵
        grad1 = grad1 + np.dot(train_images[i].T.reshape(var_num,1),
        softmax(np.dot(train_images[i].reshape(1, var_num),W1)) - onehot(train_labels[i]))
        # 均方误差
        softmax_xW = softmax(np.dot(train_images[i].reshape(1, var_num),W2))
        grad2 = grad2 + 2 * train_images[i].T.reshape(var_num,1)\
                .dot(softmax_xW - onehot(train_labels[i]))
    grad1 /= mini_batch_size
    grad2 /= mini_batch_size
    # 梯度下降
    W1 = W1 - learn_rate * grad1
    W2 = W2 - learn_rate * grad2
```

### 2.3 验证过程

模型训练完成后，通过计算$softmax(\bold {xW})$得到预测向量，并且找到该向量数值最大的那一维度作为预测结果即可。分别对使用交叉熵和均方误差进行预测正确的样本的计数，并最终输出准确预测的个数：

```python
# 验证
total = test_images.shape[0]
right1 = 0
right2 = 0
for i in range(total):
    ans = softmax(np.dot(test_images[i].reshape(1, var_num),W1))
    if np.where(ans == np.max(ans))[1][0] == test_labels[i]:
        right1 += 1
    ans = softmax(np.dot(test_images[i].reshape(1, var_num),W2))
    if np.where(ans == np.max(ans))[1][0] == test_labels[i]:
        right2 += 1  

print('验证集大小：',total)
print('预测正确个数（交叉熵损失）：',right1)
print('预测正确个数（均方误差）：',right2)
```

-------

### 三、 实验结果与分析

因为本次实验设计多个超参数，我尝试了在不同的超参数的设置下得到不同的实验结果。

在学习率为0.01、mini-batch大小为100时，设置不同的训练次数，在大小为10000的验证集上，得到了如下的实验结果：（因为训练样本有60000条，mini-batch大小为100时共分成600份，因此尽量将训练次数设置为600的倍数从而保证所有的训练数据都能平等地被训练到）

| 训练次数               | 0    | 100  | 300  | 600  | 1200 | 1800 | 2400 | 3000 | 3600 |
| ---------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 使用交叉熵的准确个数   | 873  | 8184 | 8462 | 8519 | 8838 | 8749 | 8692 | 8974 | 8459 |
| 使用均方误差的准确个数 | 873  | 8424 | 8671 | 8843 | 8734 | 8822 | 8778 | 8895 | 8589 |

因为用不同训练次数进行实验时每次都重新随机初始化了权重矩阵，结果可能有微小的误差，但不难看出，整体上，随着训练次数的增加，准确率也明显地增加了，也就是说训练使得线性函数分类趋近于实际结果，训练获得了效果。同时，在达到一定训练次数后，准确度并没有明显的增加，说明在此之后，权重矩阵基本收敛了。

通过比较使用交叉熵的损失函数和使用均方误差的损失函数的两种情况，尽管在本次实验中并不明显，但理论上使用交叉熵的效果会优于使用均方误差。均方误差的损失函数中存在`np.diag(softmax_xW.reshape(10)) - np.dot(softmax_xW,softmax_xW.T)`一项。如果是对单个$w_{ij}$求导，会得到$\sigma(w^{T}x_{i})$和$(1-\sigma(w^{T}x_{i}))$的乘积。而随着预测结果向着准确结果接近，二者的乘积必然接近于0，从而使得梯度非常小，训练对参数的改变微乎其微，很难起到效果。因此在分类问题中使用交叉熵作为损失函数更好。下面只进行交叉熵损失的讨论。

在mini-batch大小为100、学习次数为6000时，使用不同的学习率得到了如下结果：

| 学习率   | 0.0001 | 0.001 | 0.01 | 0.1  | 1    | 10   | 100  |
| -------- | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| 准确个数 | 8640   | 8923  | 8643 | 8911 | 8795 | 8780 | 8737 |

在我的实验结果中，不同的学习率基本没有太大影响。理论上在学习率较低时，需要更久的学习次数才能使得数据收敛。而在学习率较大时，容易在梯度最低点来回跳跃，到达不了最小值点且不能收敛。可以看到，学习率较大时，准确率反而有一定的下降，可能就是该原因引起的。

总的来说，训练一个多分类器时，要选择恰当的超参数，才能取得相对准确的结果。在选择损失函数时，因为均方误差存在梯度消失的现象，使用交叉熵作为损失函数是更好的选择。在不同的场景下选择损失函数、训练方法等，一定要依据当前场景具体问题具体分析，从众多选择中找出最优的一项来。