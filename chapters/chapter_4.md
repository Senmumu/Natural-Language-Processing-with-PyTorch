#第4章· 自然语言处理的前馈网络

在第3章中，我们通过观察可以存在的最简单的神经网络感知器来介绍神经网络的基础。感知器的历史性垮台之一是它无法学习数据中存在的适度非平凡模式。例如，查看图4-1中绘制的数据点。这相当于任一或（XOR）情况，其中决策边界不能是单条直线（也称为线性可分）。在这种情况下，感知器失败。“Example: Surname Classification with an MLP”.”

Excerpt From: Delip Rao. “Natural Language Processing with PyTorch.” Apple Books. 

图4-1。 XOR数据集中的两个类绘制为圆形和星形。请注意，没有一行可以分隔这两个类。
在本章中，我们将探索一系列传统上称为前馈网络的神经网络模型。我们关注两种前馈神经网络：多层感知器（MLP）和卷积神经网络（CNN）.1多层感知器在结构上扩展了我们在第3章中研究的更简单的感知器，通过将多个感知器分组在一个层中，将多个层堆叠在一起。我们在短时间内介绍了多层感知器，并展示了它们在多类分类中的应用“

摘录自：Delip Rao。 “使用PyTorch进行自然语言处理。”Apple Books。
“本章研究的第二种前馈神经网络，即卷积神经网络，受到数字信号处理中窗口滤波器的深刻启发。通过这种窗口属性，CNN能够在其输入中学习本地化模式，这不仅使它们成为计算机视觉的主力，而且还是检测顺序数据（例如单词和句子）中的子结构的理想候选者。我们在“卷积神经网络”中探索CNN，并在“示例：使用CNN对姓氏进行分类”中演示它们的用法。
在本章中，MLP和CNN组合在一起，因为它们都是前馈神经网络，与不同的神经网络系列（递归神经网络（RNN））形成对比，后者允许反馈（或循环），使得每次计算通过以前的计算得知。在图6和图7中，我们介绍了RNN以及为什么允许网络结构中的循环是有益的。
当我们浏览这些不同的模型时，确保理解工作原理的一种有用方法是在计算数据张量时注意数据张量的大小和形状。每种类型的神经网络层

“多层感知器被认为是最基本的神经网络构建模块之一。最简单的MLP是第3章感知器的扩展。感知器将数据向量2作为输入并计算单个输出值。在MLP中，许多感知器被分组，使得单个层的输出是新的矢量而不是单个输出值。在PyTorch中，您将在后面看到，只需在线性图层中设置输出要素的数量即可完成。 MLP的另一个方面是它将多个层与每层之间的非线性组合在一起。
最简单的MLP，如图4-2所示，由三个表示阶段和两个线性层组成。第一阶段是输入向量。这是给模型的矢量。在“示例：对餐馆评论的情感进行分类”中，输入向量是Yelp评论的折叠单热表示。给定输入向量，第一个线性层计算隐藏向量 - 第二阶段“

“一个简单的例子：异或
让我们看看前面描述的XOR示例，看看感知器与MLP会发生什么。在这个例子中，我们在二元分类任务中训练感知器和MLP：识别星星和圆圈。每个数据点都是2D坐标。如果不深入了解实现细节，最终的模型预测如图4-3所示。在此图中，错误分类的数据点用黑色填充，而正确分类的数据点未填充。在左侧面板中，您可以看到感知器难以学习可以分离星形和圆形的决策边界，由填充的形状证明。然而，MLP（右图）学习了一个决策边界，可以更准确地对星星和圆圈进行分类。

图4-3。来自感知器（左）和MLP（右）的学习解决方案用于XOR问题。每个数据点的真实类别是点的形状：星形或圆形。不正确的分类用黑色填充，并且没有填写正确的分类。这些行是每个模型的决策边界。在左侧面板中，感知器

多层感知器
多层感知器被认为是最基本的神经网络构建块之一。最简单的MLP是第3章感知器的扩展.Impomtron将数据vector2作为输入并计算单个输出值。在MLP中，许多感知器被分组，使得单个层的输出是新的矢量而不是单个输出值。在PyTorch中，您将在后面看到，只需在线性图层中设置输出要素的数量即可完成。 MLP的另一个方面是它将多个层与每层之间的非线性组合在一起。
最简单的MLP，如图42所示，由三个表示阶段和两个线性层组成。第一阶段是输入向量。这是给模型的矢量。在
示例：对餐馆评论的情绪进行分类“，输入向量是Yelp评论的折叠单一表示。给定输入向量，第一个线性层计算隐藏向量 - 第二个表示阶段。隐藏的向量被称为这样，因为它是输入和输出之间的层的输出。 “层的输出”是什么意思？理解这一点的一种方法是隐藏矢量中的值是构成该层的不同感知器的输出。使用此隐藏向量，第二个线性层计算输出向量。在像对Yelp评论的情绪进行分类的二进制任务中，输出向量仍然可以是大小为1.在amulticlasssetting中，您将看到示例：SurnameClassificationwithanMLP“，输出向量的大小等于类的数量。虽然在这个例子中我们只显示一个隐藏的矢量，但是可以有多个中间阶段，每个阶段产生自己的隐藏矢量。始终，使用线性层和非线性的组合将最终隐藏矢量映射到输出矢量。
                             一个“FC
  图42。 MLP的可视化表示，具有两个线性层和三个表示阶段 - 输入向量，隐藏向量和输出向量。
MLP的强大之处在于添加第二个线性层并允许模型学习可线性分离的中间表示 - 表示的属性，其中可以使用单个直线（或更一般地，超平面）来区分数据它们落在哪一侧（或超平面）的点。学习具有特定属性的中间表示，如对于分类任务可线性分离，是使用神经网络的最深刻的后果之一，并且是其建模能力的典型。在下一节中，我们将更深入，深入地研究这意味着什么。
一个简单的例子：异或
让我们看看前面描述的XOR示例，看看感知器与MLP会发生什么。在这个例子中，我们在二元分类任务中训练感知器和MLP：识别星星和圆圈。每个数据点都是2D坐标。在没有深入了解实现细节的情况下，最终的模型预测如图43所示。在此图中，错误分类的数据点用黑色填充，而正确分类的数据点未填充。在左侧面板中，您可以看到感知器难以学习可以分离星形和圆形的决策边界，由填充的形状证明。然而，MLP（右图）学习了一个决策边界，可以更准确地对星星和圆圈进行分类。
           F

  图43。来自感知器（左）和MLP（右）的学习解决方案用于XOR问题。每个数据点的真实类别是点的形状：星形或圆形。不正确的分类用黑色填充，并且没有填写正确的分类。这些行是每个模型的决策边界。在左侧面板中，感知器学习的决策边界无法正确地将圆与星星分开。事实上，没有一条线可以。在右侧面板中，MLP已经学会将星星与圆圈分开。
虽然在图中看来MLP有两个决策边界，这就是它的优势，但它实际上只是一个决策边界！决策边界就是这样出现的，因为中间表示已经变形空间以允许一个超平面出现在这两个位置中。在图44中，我们可以看到由MLP计算的中间值。点的形状表示类（星形或圆形）。我们看到的是神经网络（在这种情况下是一个MLP）已经学会“扭曲”数据所在的空间，这样它就可以在数据集通过最后一层时用一条线划分数据集。
图44。 MLP的输入和中间表示。从左到右：（1）网络输入，（2）ou

在PyTorch中实现MLP
在上一节中，我们概述了MLP的核心思想。在本节中，我们将介绍PyTorch中的实现。如上所述，MLP还有一个额外的计算层，超出了我们在第3章中看到的更简单的感知器。在我们在xample 41中提出的实现中，我们用两个PyTorch的Linear模块实例化了这个想法。线性对象命名为fc1和fc2，遵循一般惯例，将线性模块称为“完全连接层”或简称“fc层”。除了这两个线性层，还有一个整流线性单元（ReLU）非线性（在第3章，激活函数中引入），在第一线性层作为第二线性层的输入之前应用于第一线性层的输出。由于图层的顺序性，您必须注意确保图层中的输出数等于下一图层的输入数。在两个线性层之间使用非线性是必不可少的，因为没有它，序列中的两个线性层在数学上等同于单个线性层4，因此无法模拟复杂模式。我们对MLP的实现只实现了反向传播的正向传递。这是因为PyTorch会根据模型的定义和正向传递的实现自动计算出如何进行反向传递和渐变更新。
例41。使用PyTorch的多层感知器
                                                         
import torch.nn as nn
导入torch.nn.functional为F
class MultilayerPerceptron（nn.Module）：
def __init __（self，input_dim，hidden_​​dim，output_dim）：
“”Args：
input_dim（int）：输入向量的大小
hidden_​​dim（int）：第一个线性图层output_dim（int）的输出大小：第二个线性图层的输出大小
“””
super（MultilayerPerceptron，self）.__ init __（）self.fc1 = nn.Linear（input_dim，hidden_​​dim）self.fc2 = nn.Linear（hidden_​​dim，output_dim）
def forward（self，x_in，apply_softmax = False）：
“”MLP的前进传球
“EC
在例子42中，我们实例化MLP。由于MLP实现的一般性，我们可以对任何大小的输入进行建模。为了演示，我们使用大小为3的输入维度，大小为4的输出维度和大小为100的隐藏维度。请注意，如果在print语句的输出中，每个层中的单元数很好地排列以产生一个尺寸为4的输入的尺寸为4的输出。
实施例42。 MLP的示例实例化


并通过迭代字符串输入中的每个字符来创建输入的折叠单一向量表示。我们为以前没有遇到的字符指定一个特殊标记UNK。 UNK符号仍然用在字符词汇表中，因为我们仅从训练数据中实例化词汇表，并且验证或测试数据中可能存在唯一字符。 4
                                                                                   您应该注意，尽管我们在此示例中使用了折叠的单拍表示，但您会这样做
 ＃实现与示例314几乎相同
def __getitem __（self，index）：
row = self._target_df.iloc [index] surname_vector = \
self._vectorizer.vectorize（row.surname）nationality_index = \
self._vectorizer.nationality_vocab.lookup_token（row.nationality）
return {'x_surname'：surname_vector，'y_nationality'：nationality_index}
 1“ EN

在后面的章节中了解其他矢量化方法，这些方法是替代，有时甚至是单独编码。特别是在例如：ClassificationSurnamesbyUsingaCNN“中，您将看到一个单一矩阵，其中每个字符都是矩阵中的一个位置，并且有一个热点向量。然后，在第5章中，您将了解嵌入层，返回整数向量的向量化，以及如何使用这些向量来创建密集向量矩阵。但是现在，让我们来看看xample 46中SurnameVectorizer的代码。
例46。实现SurnameVectorizer
                        
class SurnameVectorizer（object）：
msgstr“”“协调词汇表并将它们用于使用的Vectorizer”
def __init __（self，surname_vocab，nationality_vocab）：self.surname_vocab = surname_vocab self.nationality_vocab = nationality_vocab
def vectorize（self，surname）：
“”将所提供的姓氏矢量化
ARGS：
姓（str）：姓氏
返回：
one_hot（np.ndarray）：折叠的onehot编码
“””
vocab = self.surname_vocab
one_hot = np.zeros（len（vocab），dtype = np.float32）姓氏中的令牌：
one_hot [vocab.lookup_token（token）] = 1返回one_hot
@classmethod
def from_dataframe（cls，surname_df）：
msgstr“”“从数据集数据框中实例化矢量化器
ARGS：
surname_df（pandas.DataFrame）：姓氏数据集
返回：
SurnameVectorizer的一个实例
“””
surname_vocab =词汇（unk_token =“@”）nationality_vocab =词汇（add_unk = False）
对于index，在surname_df.iterrows（）中的行：对于row.surname中的字母：
surname_vocab.add_token（letter）nationality_vocab.add_token（row.nationality）
return cls（surname_vocab，nationality_vocab）
 
  
 
  
  
  
 
“
  
SurnameClassifier模型
TheSurnameClassifier（xample47）是本章前面介绍的MLP的实现。第一个线性层将输入矢量映射到中间矢量，并将非线性应用于该矢量。第二线性层将中间矢量映射到预测矢量。

在最后一步中，可选地应用softmax函数以确保输出总和为1;也就是说，它被解释为“概率”.5它是可选的原因与数学公式有关，即在损失函数中引入的thecrossentropyloss“thecrossentropyloss”。回想一下，对于多类分类来说，交叉熵损失是最理想的，但是在训练期间计算softmax不仅浪费，而且在许多情况下也不是数值稳定的。
例47。使用MLP的SurnameClassifier
                          
import torch.nn as nn
导入torch.nn.functional为F
class SurnameClassifier（nn.Module）：
“”用于分类姓氏的2层多层感知器“”“
def __init __（self，input_dim，hidden_​​dim，output_dim）：
“”Args：
input_dim（int）：输入向量的大小
hidden_​​dim（int）：第一个线性图层output_dim（int）的输出大小：第二个线性图层的输出大小
“””
super（SurnameClassifier，self）.__ init __（）self.fc1 = nn.Linear（input_dim，hidden_​​dim）self.fc2 = nn.Linear（hidden_​​dim，output_dim）
def forward（self，x_in，apply_softmax = False）：
“”分类器的正向传递
ARGS：
x_in（torch.Tensor）：输入数据张量
x_in.shape应该是（batch，input_dim）apply_softmax（bool）：softmax激活的标志
如果与crossentropy损失一起使用，则应为false返回：
由此产生的张量。 tensor.shape应该是（batch，output_dim）。 “””
intermediate_vector = F.relu（self.fc1（x_in））prediction_vector = self.fc2（intermediate_vector）
如果apply_softmax：
prediction_vector = F.softmax（prediction_vector，dim = 1）
return prediction_vector
 
  
 
 
 
  
  
 
 
 
训练套路
虽然我们在此示例中使用了不同的模型，数据集和损失函数，但训练例程仍与上一章中描述的相同。因此，在示例48中，我们仅示出了在示例和示例中的训练程序中的argsandthemajordifferences：
lassifying Sentiment of Restaurant评论“。
例48。基于MLP的Yelp审阅分类器的超参数和程序选项
                           
args =命名空间（
＃数据和路径信息
1“ EC
培训中最显着的差异与模型中的输出类型和使用的损失函数有关。在此示例中，输出是可以转换为概率的多类预测向量。可用于此输出的损失函数仅限于CrossEntropyLoss（）和NLLLoss（）。由于其简化，我们使用CrossEntropyLoss（）。
在示例49中，我们展示了数据集，模型，损失函数和优化器的实例化。这些实例应该与第3章中的示例几乎完全相同。实际上，这个模式将在本书后面的章节中重复每个示例。
例49。实例化数据集，模型，损失和优化程序
                      
dataset = SurnameDataset.load_dataset_and_make_vectorizer（args.surname_csv）vectorizer = dataset.get_vectorizer（）
classifier = SurnameClassifier（input_dim = len（vectorizer.surname_vocab），hidden_​​dim = args.hidden_​​dim，
output_dim = len（vectorizer.nationality_vocab））classifier = classifier.to（args.device）
loss_func = nn.CrossEntropyLoss（dataset.class_weights）
optimizer = optim.Adam（classifier.parameters（），lr = args.learning_rate）
训练循环
该示例的训练循环几乎与“训练循环”中的训练循环中描述的训练循环相同，除了变量名称。具体地，示例410示出了使用不同的密钥来从batch_dict中获取数据。除了这种美容差异，训练循环的功能保持不变。使用训练数据，计算模型输出，损失和梯度。然后，我们使用渐变来更新模型。
例410。训练循环的片段