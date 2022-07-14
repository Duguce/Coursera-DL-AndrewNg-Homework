# 1. Improving-Deep-Neural-Networks
作业实现：改善深层神经网络部分

| 序号 | 项目名称 |                           备注                            |
| :--: | :------: | :-------------------------------------------------------: |
|  1   |  Week1   | 深度学习的实践层面 (Practical aspects of Deep  Learning)  |
|  2   |  Week2   |            优化算法 (Optimization algorithms)             |
|  3   |  Week3   | 超参数调试、Batch正则化和程序框架 (Hyperparameter tuning) |



------

<!-- TOC -->

- [1. Improving-Deep-Neural-Networks](#1-improving-deep-neural-networks)
  - [1.1. 深度学习的实践层面](#11-深度学习的实践层面)
    - [1.1.1. 训练，验证，测试集（Train / Dev / Test sets）](#111-训练验证测试集train--dev--test-sets)
    - [1.1.2. 偏差，方差（Bias / Variance）](#112-偏差方差bias--variance)
    - [1.1.3. 正则化（Regularization）](#113-正则化regularization)
    - [1.1.4. 归一化输入（Normalizing inputs）](#114-归一化输入normalizing-inputs)
    - [1.1.5. 梯度消失/梯度爆炸（Vanishing / Exploding gradients）](#115-梯度消失梯度爆炸vanishing--exploding-gradients)
    - [1.1.6. 神经网络的权重初始化（Weight Initialization for Deep Networks）](#116-神经网络的权重初始化weight-initialization-for-deep-networks)

<!-- /TOC -->



## 1.1. 深度学习的实践层面

### 1.1.1. 训练，验证，测试集（Train / Dev / Test sets）

🌱 **关键点：**

- 循环迭代过程的效率是决定一个项目进展的关键因素；
- 机器学习中，通常将样本分成训练集，验证集和测试集三部分；
- 对于小数据集分割标准：70%验证集，30%测试集；
- 数据集规模较大时，验证集和测试集要小于数据总量的20%或10%；
- 测试集的目的是对最终所选定的神经网络系统做出无偏估计；

### 1.1.2. 偏差，方差（Bias / Variance）

🌱 **关键点：**

- 高偏差对应欠拟合，高方差对应过拟合；
- 验证集误差大 ==> 高方差；
- 训练集误差大 ==> 高偏差；

- 初始模型训练完，如果偏差较高，可以尝试评估训练集或训练数据的性能，再则，可以尝试新的网络架构；
- 模型训练目标，尽可能找到一个低偏差，低方差的架构；

### 1.1.3. 正则化（Regularization）

🌱 **关键点：**

- 正则化可以被用于减少**方差**（过拟合）；

- L1正则化， 最终会是稀疏的，向量中有很多0，有利于压缩模型，因为集合中参数均为0，存储模型所占用的内存更少。实际上，虽然L1正则化使模型变得稀疏，却没有降低太多存储内存；
- ***L2正则化***是最常见的正则化类型，也被称为“权重衰减”；
- L2正则化通过设置lambda调整权重矩阵W的值；

- ***Dropout 正则化***通过遍历网络的每一层，并设置消除神经网络中节点的概率（Dropout可以随机删除网络中的神经网络），设置完节点概率，我们会消除一些节点，然后删除掉从该节点进出的连线，最后得到一个节点更少，规模更小的网络，然后用backprop方法进行训练；

- inverted dropout (反向随机失活)；

```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep-prob
a3 = np.multiply(a3, d3)
a3 /= keep-prob
```

- Dropout的一大缺点就是代价函数 不再被明确定义，每次迭代，都会随机移除一些节点， 如果再三检查梯度下降的性能，实际上是很难进行复查的；
- 常见正则化方法：1）L2正则化 2）随机失活（dropout）正则化 3）数据扩增 4）early stopping；

### 1.1.4. 归一化输入（Normalizing inputs）

🌱 **关键点：**

- 归一化包括两个步骤：零均值和归一化方差；
- 归一化可以加速神经网络的训练（特征都在相似范围内，而不是从1到1000，0到1的范围，而是在-1到1范围内或相似偏差，这使得代价函数优化起来更简单快速）；

### 1.1.5. 梯度消失/梯度爆炸（Vanishing / Exploding gradients）

🌱 **关键点：**

- （梯度消失/梯度爆炸 指）导数或坡度有时会变得非常大，或者非常小，甚至于以指数方式变小，这加大了训练的难度；

### 1.1.6. 神经网络的权重初始化（Weight Initialization for Deep Networks）

🌱 **关键点：**
$$
w^{[l]}= np.random.randn(shape) * np.sqrt(\frac{1}{n^{[l-1]}})
$$
