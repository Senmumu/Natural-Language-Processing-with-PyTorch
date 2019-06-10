# 第3章 神经网络的基本组成部分
本章通过介绍构建神经网络所涉及的基本思想，如激活函数，损失函数，优化器和监督训练设置，为后面的章节设置了阶段。 我们首先看一下感知器，一个单元神经网络，将各种概念联系在一起。 感知器本身是更复杂的神经网络中的构建块。 这是一种常见的模式，将在整本书中重复出现 - 我们讨论的每个架构或网络都可以在其他复杂网络中单独使用或组合使用。 当我们讨论计算图和本书的其余部分时，这种组合性将变得清晰。

# 感知器：最简单的神经网络
最简单的神经网络单元是感知器。在生物神经元之后，感知器在历史上和非常松散地建模。与生物神经元一样，有输入和输出，“信号”从输入流向输出，如图3-1所示。

图3-1。具有输入$（x）$和输出$（y）$的感知器的计算图。权重$（w）$和偏差（b）构成模型的参数。
每个感知器单元具有输入$（x）$，输出$（y）$和三个“旋钮”：一组权重$（w）$，偏置$（b）$和激活函数$（f）$。从数据中学习权重和偏差，并根据网络设计者对网络及其目标输出的直觉精心挑选激活功能。在数学上，我们可以表达如下：

$y = f(wx + b)$

通常情况下，感知器有多个输入。我们可以使用向量来表示这种一般情况。也就是说，x和w是向量，w和x的乘积用点积替换：

$y = f(wx + b)$

这里用f表示的激活函数通常是非线性函数。线性函数是其图形为直线的函数。在此示例中，$wx + b$是线性函数。因此，从本质上讲，感知器是线性和非线性函数的组合。 线性表达式$wx + b$也称为仿射变换。
例3-1展示了PyTorch中的感知器实现，它采用任意数量的输入，仿射变换，应用激活函数，并产生单个输出。


例3-1 使用PyTorch实现感知器
```python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    """ A perceptron is one linear layer """
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)
       
    def forward(self, x_in):
        """The forward pass of the perceptron
        
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, num_features)
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()
```
PyTorch在torch.nn模块中方便地提供了一个Linear类，它可以完成权重和偏差所需的簿记，并进行必要的仿射变换.1在“深入潜水监督培训”中，您将看到如何“学习”来自数据的权重w和b的值。前面例子中使用的激活函数是sigmoid函数。在下一节中，我们将介绍一些常见的激活功能，包括此功能。
# 激活函数
激活函数是在神经网络中引入的非线性，以捕获数据中的复杂关系。在“深入监督培训”和“多层感知器”中，我们深入探讨了学习中需要非线性的原因，但首先，让我们看一些常用的激活函数。
# Sigmoid
sigmoid是神经网络历史中最早使用的激活函数之一。它取任何实际值并将其压入0到1之间的范围。数学上，sigmoid函数表示如下：

$f(x) = \frac{1}{1+e^-x}$

从表达式中可以很容易地看出，S形是一种平滑，可微分的功能。PyTorch将sigmoid实现为`torch.sigmoid()`，如例3-2所示。
例3-2。 Sigmoid激活
```python
import torch
import matplotlib.pyplot as plt

x = torch.range(-5., 5., 0.1)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

```
![](../../imgs/page46image57632512.png)

正如您可以从图中看到的那样，S形函数非常快速地饱和（即产生极值输出）并且对于大多数输入而言。这可能成为一个问题，因为它可能导致梯度变为零或发散到溢出的浮点值。这些现象也分别称为消失梯度问题和爆炸梯度问题。因此，很少看到除了输出之外的神经网络中使用的sigmoid单位，其中压缩属性允许人们将输出解释为概率。
#Tanh
tanh激活函数是S形的美容上不同的变体。 当你写下tanh的表达式时，这一点就变得清晰了：

$f(x)= tanh x = \frac{e^x-e^-x}{e^x+e^-x}$

通过一些争论（我们将其作为练习留给您），您可以说服自己，tanh只是S形函数的线性变换，如例3-3所示。 当你写下tanh（）的PyTorch代码并绘制曲线时，这一点也很明显。 请注意，tanh与sigmoid一样，也是一种“挤压”函数，除了它将实际值集合从$（-∞，+∞）$映射到范围$[-1，+ 1]$。“


# RELU
ReLU 代表整流线性单元。 这可以说是最重要的激活功能。 事实上，人们可以冒险说，如果不使用ReLU，很多近期的深度学习创新都是不可能的。 对于一些如此基础的东西，就神经网络激活功能而言，它也是一个令人惊讶的新东西。 它的形式非常简单：

$f(x) = max(0,x)$

因此，所有ReLU单元做的是将负值剪切为零，如例3-4中所示。
```python
import torch
import matplotlib.pyplot as plt

relu = torch.nn.ReLU()
x = torch.range(-5., 5., 0.1)
y = relu(x)

plt.plot(x.numpy(), y.numpy())
plt.show()
```
 ![](../../imgs/page47image56741328.png)
ReLU的削波效应有助于消除梯度问题也可能成为一个问题，随着时间的推移，网络中的某些输出可能会简单地变为零，永远不会再次复活。 这被称为“垂死的ReLU”问题。 为了减轻这种影响，已经提出了诸如Leaky ReLU和Parametric ReLU（PReLU）激活函数的变体，其中泄漏系数a是学习参数。 例3-5显示了结果。
```python
import torch
import matplotlib.pyplot as plt

prelu = torch.nn.PReLU(num_parameters=1)
x = torch.range(-5., 5., 0.1)
y = prelu(x)

plt.plot(x.numpy(), y.numpy())
plt.show()
```

# Softmax
激活功能的另一个选择是softmax。 与sigmoid函数一样，softmax函数将每个单元的输出压缩到介于0和1之间，如例3-6所示。 然而，softmax操作还将每个输出除以所有输出的总和，这给出了k个可能类的离散概率分布3

$softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{k} e^{x_j}}$

所得分布的概率总和为1。 这对于解释分类任务的输出非常有用，因此这种转换通常与概率训练目标配对，例如分类交叉熵，“深入潜水监督培训”中对此进行了介绍。
- input 0
```python
import torch.nn as nn
import torch

softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)
print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))
```
- output 0
```
tensor([[ 0.5836, -1.3749, -1.1229]])
tensor([[ 0.7561,  0.1067,  0.1372]])
tensor([ 1.])
```
# Loss function
在第1章中，我们看到了一般监督机器学习架构以及损失函数或目标函数如何帮助指导训练算法通过查看数据来选择正确的参数。 回想一下，损失函数将真值（y）和预测（ŷ）作为输入并产生实值得分。 该分数越高，模型的预测就越差。 PyTorch在其nn包中实现了比我们可以覆盖的更多的损失函数，但我们将回顾一些最常用的损失函数。

-input 0
```
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)
```
- output 0
```
tensor([[ 0.5836, -1.3749, -1.1229]])
tensor([[ 0.7561,  0.1067,  0.1372]])
tensor([ 1.])
```
在本节中，我们研究了四个重要的激活函数：sigmoid，tanh，ReLU和softmax。 这些只是您可以用于构建神经网络的许多可能激活中的四种。 随着我们逐步完成这本书，将会清楚应该使用哪些激活功能以及在哪里，但一般指南只是简单地遵循过去的工作。
## “Mean Squared Error Loss

对于网络输出（ŷ）和目标（y）是连续值的回归问题，一个常见的损失函数是均方误差（MSE）：


