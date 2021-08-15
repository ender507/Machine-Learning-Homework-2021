# 机器学习与数据挖掘

# Assignment 2实验报告

>姓名：507
>
>学号：18340057
>
>班级：计算机科学二班

## 一、 理论知识

### 1.1 SVM 模型的基本理论  

在之前的课程中讨论的分类器都是线性的，而在实际问题中，很多数据并不是线性可分的，也就是说找不到这样的超平面，能完全区分不同的数据。所以，需要在分类器中引入非线性成分，使得模型更好地贴合数据，让分类更加准确。为了使模型非线性化，我们可以通过基函数将原始特征x变换到另一个空间。问题变为了原始最大裕度优化问题：
$$
\min_{\bold w,b,\bold \xi}\frac12||\bold w||^2+C\sum^N_{n=1}\xi_n\\
s.t.:y^{(n)}\cdot(\bold w^T\bold \phi(\bold x^{(n)})+b)\geq1-\xi_n
$$
得到的分类器为：
$$
\hat y(\bold x)=sign(\bold w^{*T}\phi(\bold x^{(n)})+b^*)
$$
从直观上看，数据在高维空间中更容易分离。为了获得更好的性能，我们希望映射后的x到更高维的空间。然而太高的话代价也很大。使用对偶形式方法解决时会要计算映射值的转置与自身的内积，导致高开销。这个问题可以通过使用内核技巧来解决。

核函数是一个二元函数，可以表示为某些函数的内积：
$$
k(\bold x,\bold x')=\phi(\bold x)^T\phi(\bold x')
$$
Mercer定理：如果函数$k(\bold x,\bold x')$是对称正定的，即：
$$
\int\int g(\bold x)k(\bold x,\bold y)g(\bold y)d\bold xd\bold y\geq0\forall g(\cdot)\in L^2
$$
就存在函数$\phi(\cdot)$使得$k(\bold x,\bold x')=\phi(\bold x)^T\phi(\bold x')$。一个函数如果满足正定条件就必然是核函数。

最常用的核函数之一是高斯核，有着无限维：
$$
k(\bold x,\bold x')=\exp\{-\frac1{2\sigma^2}||\bold x-\bold x'||^2\}
$$
利用核函数，可以将对偶最大边距分类器重写为：
$$
\max_{\bold a}g(\bold a)s.t.:a_n\geq0,a_n\leq C,\sum^N_{n=1}a_ny^{(n)}=0
$$
其中，
$$
g(\bold a)=\sum^N_{n=1}a_n-\frac12\sum^N_{n=1}\sum^N_{m=1}a_na_my^{(n)}y^{(m)}k(\bold x^{(n)},\bold x^{(m)})
$$
从而得到诱导分类器：
$$
\hat y(\bold x)=sign(\sum^N_{n=1}a_n^*y^{(n)})k(\bold x^{(n)},\bold x^{(m)})+b^*)
$$
核技巧：将函数k代入。如果$\phi$不改变$\bold x$则为线性最大边际分类器，否则为基于基函数的有限维非线性最大边际分类器，如果为高斯核，则为无限维非线性最大边际分类器。

### 1.2 hinge loss线性分类和SVM模型之间的关系

活页损失hinge loss定义如下：
$$
L(\bold w)=\max(0,1-yh)
$$
其中$h=\bold w^T\bold x$，$y=\pm1$。从公式可以直观地看出，当实际标签为1时，预测结果小于1则会产生梯度，否则梯度为0。而实际标签为-1时，预测结果大于-1时会产生梯度，否则梯度为0。也就是说，在活页损失下，该模型会自动舍弃一些预测结果正确的样本。即便这些样本也会产生误差，但它们都不是误差的主要来源。该模型下只会考虑带来更大误差的样本，这和最大边际分类器的思想是一致的。最大边际分类器的要求是找到两个类中距离超平面最近的两点，将这个距离和作为边际，并将其最大化。也就是说，只考虑离超平面最近的点来修正超平面的位置，而不考虑离超平面更远的样本。

一般的线性最大边际分类器的条件为：
$$
\min_{\bold w,b,\bold \xi}\frac12||\bold w||^2s.t.:y^{(n)}\cdot(\bold w^T\bold x^{(n)}+b)\geq1
$$
使用hinge loss的模型还可以加入松弛变量，从而弱化条件，使得非线性可分的数据具有一定的容忍性：
$$
\min_{\bold w,b,\bold \xi}\frac12||\bold w||^2+C\sum^N_{n=1}\xi_ns.t.:y^{(n)}\cdot(\bold w^T\bold x^{(n)}+b)\geq1-\xi_n
$$
但是即便如此，使用hinge loss的线性分类模型仍然是线性的。SVM模型在此基础上，将$\bold x$通过一个基函数$\phi$变换到另一个空间，从而真正实现了非线性成分的引入。问题变为了：
$$
\min_{\bold w,b,\bold \xi}\frac12||\bold w||^2+C\sum^N_{n=1}\xi_n\\
s.t.:y^{(n)}\cdot(\bold w^T\bold \phi(\bold x^{(n)})+b)\geq1-\xi_n
$$
可以看到，使用hinge loss的线性分类模型和SVM模型的区别在于，原来hinge loss线性分类模型中的$\bold x$的项全部被换成了$\phi(\bold x)$，从而实现了$\bold x$到高维空间的映射，使得模型获得了非线性成分。

-----------

## 二、 训练过程

### 2.1 线性和高斯核函数的SVM的实现

首先读入数据集：

```python
train_images = np.load("train-images.npy")
train_labels = np.load("train-labels.npy")
test_images = np.load("test-images.npy")
test_labels = np.load("test-labels.npy")
print("验证集大小：",test_images.shape[0])
```

我使用了python的`sklearn`库来实现SVM：

```python
from sklearn import svm
```

以线性核函数的SVM模型为例，首先创建模型：

```python
model = svm.SVC(kernel='linear') 
```

然后进行训练：

```python
model.fit(train_images, train_labels)
```

训练时可以调整一定的超参数。对于线性分类器来说，需要确定模型训练结束的标志，可以设置模型精度和最大迭代次数。这里我选用库里的默认参数，即不设置最大迭代次数，模型精度为`tol = 1e-3`。模型会自适应地决定每个类所占据的权重，不同的类设置不同的惩罚参数。

训练后进行预测与结果统计：

```python
predicted= model.predict(test_images)# 预测
# 统计
right = 0
for i in range(test_images.shape[0]):
    if predicted[i] == test_labels[i]:
        right += 1
print("分类正确个数（线性核）：",right)
print("准确率（线性核）：",right/test_images.shape[0])
```

高斯核的代码也基本一致，在创建模型时将核函数的参数改为高斯核`rbf`(径向基函数)即可：

```python
model = svm.SVC(kernel='rbf') 
```

高斯核中需要确定$\sigma$的参数值，在`sklearn`的参数中为$gamma$，其中$gamma=\frac1{2\sigma^2}$。我的实现中仍采用默认值，即$gamma$值为特征值个数的倒数。

### 2.2 hinge loss 和 cross-entropy loss 的线性分类模型实现

#### 2.2.1 hinge loss线性分类模型

 在读入数据后需要注意，使用合页损失hinge loss的话，要求二分类的标签为1或-1，而不是之前的1或0。所以需要更改标签：

```python
train_labels[train_labels==0] = -1
test_labels[test_labels==0] = -1
```

对每个特征向量，还要增加一维1作为偏移项。在本次的实验中，处理的图像都属于图片，像素值范围为0~255，因此可以考虑将所有原有的特征值除以255进行归一化处理。即：
$$
\forall i=1,2,...,N\quad x_i=\frac{x_i-x_{min}}{x_{max}-x_{min}}
$$
从而使得所有特征值的范围为$[0,1]$，便于之后的梯度计算：

```python
tmp = np.zeros((train_images.shape[0],train_images.shape[1]+1))
for i in range(train_images.shape[0]):
    tmp[i] = np.append(train_images[i]/255, 1)
train_images = tmp[:][:]
tmp = np.zeros((test_images.shape[0],test_images.shape[1]+1))
for i in range(test_images.shape[0]):
    tmp[i] = np.append(test_images[i]/255, 1)
test_images = tmp[:][:]
```

然后设置基本的变量和超参数：

```python
train_num = train_images.shape[0]   # 训练集大小
var_num = train_images.shape[1]     # 训练样本特征数
w = np.random.rand(var_num)         # 随机初始化权值向量
learn_rate = 0.001                  # 学习率
train_times = 1000                  # 训练次数
```

在训练过程中，需要对损失函数求梯度。hinge loss为：
$$
L(\bold w)=\max(0,1-yh)
$$
其中$h=\bold w^T\bold x$，$y=\pm1$。

也就是说，当$1-y(\bold w^T\bold x)$小于等于0时，梯度就为0。换言之，当前的样本对修正梯度没有任何影响，直接忽略了。否则，对合页损失函数求导得到$-hx$，即：
$$
\frac{\part L(\bold w)}{\part \bold w}=\left\{\begin{array}{}0\quad1-y(\bold w^T\bold x)\leq0\\-\bold w^T\bold x\cdot\bold x\quad otherwise\end{array}\right.
$$
从而可以使用梯度下降法对模型进行训练：

```python
# 训练
for t in range(train_times):
    # 初始化梯度为0 
    grad = np.zeros((var_num))
    # 计算梯度
    for i in range(train_num):
        if 1 - np.dot(train_images[i],w) * train_labels[i] < 0:          
            continue
        grad = grad - (np.dot(train_images[i],w) 
                       * train_labels[i]) * train_images[i]
    grad /= train_num
    # 梯度下降
    w = w - learn_rate * grad
```

验证时判断$\bold w^T\bold x$的符号即可。若为正则预测结果为1，否则为-1：

```python
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
```

#### 2.2.2 cross-entropy loss 线性分类模型

使用交叉熵损失训练二分类模型需要用到`sigmoid`函数。该函数为：
$$
\sigma(x)=\frac1{1+e^{-x}}
$$
$e^{-x}$将$x$映射到$(0,+\infin)$，在实际运算过程中，$x$数值过小时可能会造成溢出。因此考虑将$x$分符号讨论，保证$-x$为负，使得映射的结果为$(0,1]$：
$$
\sigma(x)=\left\{\begin{array}{}\frac1{1+e^{-x}},x\geq0\\
\frac{e^x}{1+e^x},x<0\end{array}\right.
$$
在数值角度来说没有任何变化，但实际运算过程中能有效避免数值溢出：

```python
def sigmoid(x):
    x = float(x)
    if x<0:
        ans = np.exp(x)/(1+np.exp(x))
    ans = 1/(1+np.exp(-x))
    return ans
```

但是需要注意的是，`sigmoid`函数仍然对数值要很强的要求。当上述函数的输入参数`x​`到达-10及以下时，结果会非常接近0，导致计算梯度时造成梯度消失。为了避免绝对值较大的数值的出现，我希望初始化时，$\bold w$和$\bold x$的点乘结果的绝对值不会大于1，所以进行了以下操作：

- 将所有$\bold x$的值除以最大特征值255。同时，增加偏移项时，将加在$\bold x$内的1也除以255，这样一来就能保证所有的特征值都在$[0,1]$中：

```python
tmp = np.zeros((train_images.shape[0],train_images.shape[1]+1))
for i in range(train_images.shape[0]):
    tmp[i] = np.append(train_images[i], 1)/255
train_images = tmp[:][:]
tmp = np.zeros((test_images.shape[0],test_images.shape[1]+1))
for i in range(test_images.shape[0]):
    tmp[i] = np.append(test_images[i], 1)/255
test_images = tmp[:][:]
```

另外，初始化权重向量时也要注意。如果所有的$\bold w$中的值都在$[-1,1]$中，的确可以保证$\bold w$和$\bold x$两个向量的对应元素一一相乘时绝对值都不会大于1，但是将这些元素加和起来后绝对值又可能会大于1。特征向量维度越大越可能。因此，初始化$\bold w$，使得所有值都在$[-1,1]$中后，还要进一步除以特征向量的维度数，从而保证点乘结果的绝对值不会大于1。

同时，也要设置较小的学习率，防止$\bold w$变化太大，又产生之前提及过的`sigmoid`函数产生梯度消失的结果：

```python
# 设置基本变量和超参数
train_num = train_images.shape[0]   # 训练集大小
var_num = train_images.shape[1]     # 训练样本特征数
w = np.random.rand(var_num)/var_num # 随机初始化权值向量
learn_rate = 0.001                  # 学习率
train_times = 500                   # 训练次数
```

之后的训练和验证过程与上面的hinge loss线性分类器基本一致，不再赘述。只不过交叉熵损失下，计算梯度的公式要改为：
$$
\frac{\part L(\bold w)}{\part\bold w}=\frac1N\sum^N_{l=1}[\sigma(\bold {x}^{(l)}\bold w)-y^{(l)}]\bold x^{(lT)}
$$
其中$\sigma$表示`sigmoid`函数。即：

```python
grad = grad + (sigmoid(np.dot(train_images[i],w)) 
               - train_labels[i]) * train_images[i] 
```



-----

## 三、 实验结果与分析

### 3.1 线性和高斯核函数的SVM

此处结果来自打包程序的`SVM.py`。

验证集大小为2115，而使用线性核分类正确的个数为2113，使用高斯核分类正确的个数为2114。可以看出，使用SVM的效果较好，绝大部分的数据都能进行有效地分类。因为高斯核将特征向量映射到无穷花维，使用高斯核理论上可以使用超平面将全部的数据完全地分离，从而实现完全准确的预测。但因为精度限制较为宽松，所以没有完全分开。本次实验中高斯核的效果也优于使用线性核的SVM。因为本次实验给出的数据集基本上线性可分，因此结果差别不大，但因为高斯核引入了非线性成分，理论上性能可以优于线性核。

### 3.2 hinge loss 和 cross-entropy loss 线性分类模型的比较  

此处结果来自打包程序的`hinge.py`和`cross-entropy`。

使用hinge loss的结果如下，其中，没有标注的数字表示已经训练的次数。

<img src="pic\\1.png" style="zoom:33%;" />

使用cross-entropy loss的结果如下：

<img src="pic\\2.png" style="zoom: 50%;" />

二者的学习率相同，都为0.001，而使用交叉熵损失的模型的偏移量`b`比使用合页损失的模型更小（上面的实现部分已经具体讨论过），但是可以发现，使用交叉熵损失的模型的收敛速率明显快于使用合页损失的模型。这是因为在合页损失下，模型会筛除很多分类正确但仍存在误差的样本。这些样本也存在误差，但在合页损失下不提供任何梯度。只有那些误分类和离分类边界很近的样本才会产生梯度用于模型的训练；而交叉熵损失函数会考虑每一个样本的误差。如此下来，交叉熵损失的线性模型的收敛速度就明显快于使用合损失的模型。

本次实验中的数据高度线性可分，因此使用交叉熵损失的模型的结果也较好，在2115个验证集样本中，预测正确了2111个。这是上面提及的考虑每个样本的误差来计算梯度造成的。但是，使用合页损失的线性模型也有自己的优势。正因为合页损失不会考虑到很多正确分类的样本而是考虑误分类的样本和正确分类但离分类边界很近的样本，这为合页损失的线性分类模型提供了一定的泛化能力。在面对较难用线性分类器区分的数据、分类边界较为模糊时，使用合页损失往往能有更好的性能表现。

