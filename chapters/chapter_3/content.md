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