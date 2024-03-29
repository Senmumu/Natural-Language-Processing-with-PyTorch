O'Reilly的标志是O'Reilly Media公司自然语言的注册商标。
处理与PyTorch，封面图片，以及相关的商业外观是奥赖利的商标。
媒体公司
在这项工作中所表达的观点仅代表作者本人意见，不代表出版商
观点。虽然出版商和作者们用真诚的努力，以确保
信息和包含在此工作的指示是准确的，出版商和作者
拒绝的错误或遗漏承担任何责任，包括但不限于对责任
因使用或依赖这些工作所造成的损害。的信息的使用和
包含在此工作的指示是在你自己的风险。如果有任何代码示例或其它技术
这项工作包含或描述受开放源代码许可或知识产权
他人的权利，这是你的责任，以确保其使用这样的规定
许可证和/或权限。

# 前言
这本书的目的是使初来乍到的自然语言处理（NLP）和深学习到
品酒表涵盖重要议题在这两个领域。这两个学科领域正在增长
成倍。因为它引入了两个深刻的学习和NLP与重点
实施，这本书中占有重要的中间地带。在写这本书，我们
不得不做出什么材料，离开了困难，有时不舒服，选择。对于
初学者的读者，我们希望这本书将提供基础，并窥见了坚实的基础
什么是可能的。机器学习，特别是深度学习，是一种体验
纪律，而不是一种智力的科学。在每个慷慨端到端代码示例
章邀请您来的经验分享。
当我们开始完成本书的工作，我们开始PyTorch 0.2。这些例子进行了修订
每个PyTorch更新，从0.2到0.4。
PyTorch 1.0是因为当这本书，约释放
出来。书中的代码示例PyTorch 0.4兼容的，并应当在工作，他们
与即将到来的PyTorch 1.0版本。
关于这本书的风格的注释。我们已经有意避免数学中最的地方，不是因为深度学习数学是特别困难的（它不是），而是因为它是一个
在许多情况下分心本的主要目标书，授权初学者学习者。同样，在许多情况下，无论是在代码和文本，我们有超过青睐博览会
简洁。先进的读者和有经验的程序员可能会看到的方式来收紧
代码等，但我们的选择是尽可能明确，以达到最广泛的
我们想要覆盖的受众群体。
# 本书中使用的约定
以下印刷约定本书中使用：
斜体
表示新术语，URL，电子邮件地址，文件名和文件扩展名

恒宽
用于程序清单，以及段落中引用程序元素如
变量或函数名，数据库，数据类型，环境变量，语句和
关键字。
等宽粗体
显示命令或应该从字面上用户键入其他文本。
等宽斜体
可见，应与用户提供的值或由确定的值来代替文本
上下文。
小费
该元素表示一个尖端或建议。
注意
此元素表示一般的注释。
警告
这个元素表示警告或警告。
使用代码示例
补充材料（代码示例，练习等）可供下载
H
载荷大小：//nlproc.info/PyTorchNLPBook/repo/。
这本书的目的是帮助您完成您的工作。在一般情况下，如果示例代码是提供与此
本书中，你可以在你的程序和文档中使用它。您不必与我们联系
许可除非你复制代码的显著部分。例如，写

编写使用本书中几个代码块的程序不需要许可。出售或
分配的例子来自O'Reilly出版的一光盘则需要获得许可。回答
引用本书和报价示例代码并不需要许可的问题。
将本书的例子代码显著量到你的产品
文档则需要获得许可。
我们赞赏，但不要求，归属。归属通常包括标题，作者，
出版商和ISBN。例如：“自然语言处理与PyTorch由Delip饶
和布莱恩·麦克马汉（O'Reilly出版）。版权所有2019年，Delip Rao和布赖恩·麦克马汉，9781491
978238.”
如果你觉得你的代码示例使用落在外面正当使用或上面给出的许可，感觉
自由与我们联系
p
ermissions@oreilly.com。
O'Reilly Safari
小号
afari（原野生动物园联机丛书）是membershipbased培训和参考平台
为企业，政府，教育，和个人。
会员有机会获得成千上万的书籍，培训录像，学习路径，互动
教程，以及来自250出版商，包括O'Reilly Media公司，哈佛策划播放列表
商业评论，Prentice Hall的专业，艾迪生韦斯利专业，微软出版社，
萨姆斯，阙，购买PeachpitPress出版社，Adobe公司，焦点新闻，思科出版社，John Wiley和Sons，Syngress，
摩根考夫曼，IBM红皮书年底Packt，Adobe公司的新闻，FT出版社，Apress出版，曼宁，新
骑手，McGrawHill，琼斯和巴特利特，当然技术，等等。
欲了解更多信息，请访问：
H
TTP：//oreilly.com/safari。
如何联系我们
请处理有关这本书的出版商的意见和问题：
O'Reilly Media公司
1005格拉芬施泰因高速公路北
塞瓦斯托波尔，CA 95472
8009989938（在美国或加拿大）
7078290515（国际或本地）


# 第1章简介
像回声（Alexa的），Siri的，和谷歌翻译家喻户晓的名字有至少一个共同的事情。
他们是来自自然语言处理（NLP）的应用衍生的所有产品，的一个
这本书的两大主题。 NLP是指涉及应用一组技术
统计方法，有或没有从语言学的见解，了解课文解决的缘故
真实世界的任务。本文的“理解”主要衍生通过将文本，以可用
计算表示，这是离散的或连续的组合结构如
载体或张量，图表和树木。
适用于从数据的任务（文本在这种情况下）表示的学习是机器的主题
学习。机器学习为文本数据的应用程序有超过三个十年的历史，
但是在过去的10年里，一系列被称为深度学习的机器学习技术仍在继续
发展和开始，以证明在NLP各种人工智能（AI）的任务非常有效，
语音和计算机视觉。深度学习是我们覆盖的另一个主要对象;因此，这本书是一
NLP和深入研究学习。
# 注意
引用在本书每一章的末尾列出。
简单地说，深度学习使人们能够有效地利用数据表示学习
所谓抽象的计算图表和数值优化技术。这就是成功
深学习和计算图表，各大高科技公司，如谷歌，Facebook和
计算图形框架和库的亚马逊出版的实现建立在
他们捕捉到研究人员和工程师的心理份额。在这本书中，我们考虑PyTorch，一
日益流行的Pythonbased计算图形框架实现深度学习
算法。在本章中，我们解释什么是计算图是我们选择使用PyTorch的
作为框架。
机器学习和深度学习的领域是广阔的。在这一章中，对于大多数本书中，我们
主要是考虑什么叫做监督学习;也就是说，与训练样本的学习。我们
解释监督学习模式，将成为这本书的基础。如果你不
熟悉许多这些条款至今，你在正确的地方。本章，未来沿
章，不仅澄清也将深入到他们。如果你已经熟悉了一些
术语和概念在这里所提到的，我们还是鼓励你跟着，原因有二：
建立一个共享的词汇书的其余部分，并填补了解所需的任何空白
以后的章节。
本章的目标是：
制定监督学习模式有清晰的认识，理解的术语，以及
制定一个概念框架，以接近的学习任务以后的章节。

了解如何编码输入的学习任务。 了解什么是计算图都是。 掌握PyTorch的基础知识。 让我们开始吧！ 监督学习范式 在机器学习，或监督学习监督，是指情况下地面实况的 目标（什么东西被预测的）可用于观察。例如，在文件 分类，目标是分类标签，观察是文件。在机 翻译，观察是一个语言的句子，目标是在另一个句子 语言。与输入数据的这种认识，我们说明在监督式学习模式 F igure 11。 图11。监督学习范例，用于从标记的输入数据中学习的概念框架。 我们可以打破监督式学习模式，如所示 F igure 11，六个主 概念： 意见 意见是这是我们想要的东西预测项目。我们使用表示意见 X。我们有时指的是观测作为输入。 目标 靶是对应于观察的标签。这些通常被预测的事情。 继学习机/深度学习的标准符号，我们使用y以参考这些。 有时，这些标签被称为地面实况。 模型 模型是一数学表达式或接受一个观察中，x的函数，并预测 其目标标签的价值。 参数 有时也被称为权重，这些参数模型。这是标准的用符号w ^ （为权重）或W。 预测

预测，也叫估计，是由模型猜到目标的值，考虑到
观察结果。我们这些使用“帽子”符号表示。所以，靶y的预测被表示为
年。
损失函数
损失函数是比较多远关预测的功能是从它的目标
在训练数据的观测。给定一个目标和它的预测，损失函数分配一个标
真正的价值叫损失。损失的值越小，越好的模型在预测
目标。我们用L表示损失函数。
虽然这不是绝对必要的是数学上正式将在NLP生产/深
学模型或写这本书，我们将正式重申监督学习范式
装备读者谁是新的领域与标准术语，让他们有一定的了解
与符号和写作风格的研究论文，他们可能会遇到的arXiv。
考虑一个包含n个示例的数据集。鉴于此数据集，我们要学习的功能
（模型）的F由权重w参数化。也就是说，我们做出关于F的结构假设，以及
考虑到结构，权重的学习值W将充分表征模型。对于给定的
输入X，该模型预测Y作为目标：
在监督学习，培训的例子，我们知道一个观察真正的目标年。损失
此实例然后将L（Y，Y）。然后监督学习成为寻找的过程
最佳参数/权重w，将最小化对所有n个实施例中的累计损失。
培训使用（随机）梯度下降
监督学习的目标是挑选可以最大限度地减少损失函数的参数值
对于给定的数据集。换言之，这等同于在方程式中发现的根源。我们知道
梯度下降是找到一个方程的根的常用技术。回想一下，在传统
梯度下降，我们猜测为根（参数）一些初始值，并更新
参数迭代，直到目标函数（损失函数）的计算结果为低于一个值
可接受的阈值（也称为收敛准则）。对于大型数据集，实现传统
在整个数据集中梯度下降通常是不可能的，由于存储器的限制，而且很
减缓由于计算成本。相反，梯度下降逼近叫
随机梯度下降（SGD）通常采用。在随机的情况下，一个数据点或
数据点的子集，随机挑选，并在梯度计算为该子集。当一个
单个数据点时，该方法被称为纯SGD，而当一个子集（一个以上的）
正在使用的数据点，我们称其为minibatch SGD。通常的话“纯粹”和“minibatch”是
当所使用的方法是明确的基于上下文下降。在实践中，纯粹是新元
很少使用，因为它会导致非常慢的收敛由于嘈杂的更新。有不同
一般SGD算法的变种，都瞄准了更快的收敛。在后面的章节中，我们
探索一些这些变体与梯度如何在更新所述参数用于沿。
迭代更新的参数，这个过程被称为反向传播。每一步（又名
反向传播的历元）由直传和反向通。直传
评估与该参数的当前值的输入，并计算损失函数。该
向后通更新使用损失的梯度的参数。
注意，到现在为止，这里没有什么是特定的深度学习或神经网络。方向
在箭头的
F
igure 11指示数据的“流”而训练系统。我们将有更多的
说的培训和“流”的概念
“
计算图”，但首先，让我们来看看
我们如何能够代表我们的投入和目标，NLP问题的数值，这样我们可以训练

但首先，让我们来看看
我们如何能够代表我们的投入和目标，NLP问题的数值，这样我们可以训练
D = {，} X i y i
ñ
I = 1
ŷ= f（X，w）
3
模型和预测结果。
观察和目标编码
我们需要数字代表的意见（文本）与机器一起使用它们
学习算法。
F
igure 12呈现的视觉描绘。
图12。观察和目标编码：从目标和意见
F
igure 11被表示
数字表示为向量或张量。这被统称为输入“编码”。
来表示文本的简单方法是作为一个数值向量。有无数的方法来执行此
映射/表示。事实上，很多这本书是专门学习这种表述为
从任务数据。但是，我们首先是基于一些简单的countbased交涉
启发式。虽然简单，他们是因为他们是令人难以置信的强大，可以作为一个起点
更丰富的表现学习。所有这些countbased交涉开始的固定载体
尺寸。
一热表现
的onehot表示，顾名思义，开始于零矢量，并设置为1的
如果字存在于句子或文件对应于载体中的条目。考虑
下面两个句子：
时光飞逝像一个箭头。
果蝇像香蕉。
符号化的句子，忽略标点符号和处理一切为小写，将产生
大小为8的词汇：{时间，水果，苍蝇，等等，一，一个箭头，香蕉}。所以，我们
可以代表与eightdimensional onehot向量中的每个单词。在本书中，我们用1来表示
对于令牌/单词w onehot表示。
一个短语，句子，或文档中倒塌onehot表示是一个简单的逻辑或
其构成词的onehot表示。使用中所示的编码
F
igure 13，一个
为“像香蕉”这一短语热表示将一个3×8矩阵，其中，所述列是
eightdimensional onehot载体。这也是常见的看到一个“塌陷”或二进制编码，其中
文本/短语由向量表示的词汇表的长度，用0和1来表示
不存在或一个字的存在。为“像香蕉”二进制编码然后将：[0，0，0，
1，1，0，0，1]。

图13。用于编码的句子one-hot编码：“Time flies like an arrow”和“Fruit flies like a
banana.”

注意
在这一点上，如果你不快地发现了我们混淆了两种不同的含义（或感觉）
“苍蝇”，祝贺的，聪明的读者！语言是充满了暧昧的，但我们仍然可以
通过使惊人的简化假设，建立有用的解决方案。它可以学习
特定场景的编码，但我们现在正在超越自我的。

虽然我们很少会在这本书中使用其他任何比one-hot表示用于输入，
现在我们将介绍*Term-Frequency*（TF）和 *Term-Frequency-Inverse-Document-Frequency*
（TFIDF）表示。这样做是因为他们在NLP普及，由于历史的原因，和
为了完整起见。这些表示在信息检索历史悠久（IR）
并积极采用即使在今天，在生产NLP系统。
TF表示
一个短语，句子，或文档的TF表示仅仅是onehot的总和
其构成词表示。要继续与我们的愚蠢的例子，使用上述
onehot编码，句子“果蝇像时光飞逝水果”有以下TF表示：
[1，2,2，1，1，0，0，0]。请注意，每个条目的次数数的计数
对应的词出现在句子（文集）。我们表示一个字的W进行TF（W）的TF。
实施例11。生成“坍塌”使用scikitlearn onehot或二进制表示
```python
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
corpus = ['Time flies flies like an arrow.',
          'Fruit flies like a banana.']
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
            cbar=False, xticklabels=vocab,
            yticklabels=['Sentence 2'])
```
图14。例11所产生的坍缩的one-hot编码。

TF-IDF表示
考虑专利文献的收集。你可能会认为他们中的大多数包含类似的话
根据权利要求，系统，方法，程序，等等，经常重复多次。课题组表示
权重比例的话他们的频率。然而，如“要求”常用词不加
什么我们具体专利的理解。相反，如果一个生僻字（如
"tetrafluoroethylene"(四氟乙烯)）不经常发生，但很可能会成为指示的性质
专利文件中，我们希望给它在我们表示较大的权重。逆
*Document-Frequency*（IDF）是一种启发式的做到这一点。
的IDF表示惩罚共同令牌和奖励在矢量表示罕见令牌。
令牌W是相对于定义一个语料库为一体的IDF（W）：
其中n是包含单词w的文档的数量，N是文档的总数。
该TFIDF分数仅仅是产品的TF（W）* IDF（W）。首先，请注意，如果有一个很常见
在所有文件中出现的单词（即，n = N），IDF（w）为0并且TFIDF得分为0，从而
完全惩罚这个词。其次，如果出现长期很少，只有一个文件也许，
以色列国防军将是最大可能值，登录N.

xample图12示出如何生成TFIDF
使用scikitlearn英语句子的列表的表示。


例1-2 使用scikit-learn生成一个TF-IDF表示
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab,
            yticklabels= ['Sentence 1', 'Sentence 2'])
```

在深度学习，这是难得一见的输入采用启发式表示喜欢TFIDF编码的，因为
我们的目标是学习的表示。通常情况下，我们开始了onehot编码使用整数指数和
一个特殊的“嵌入查找”层来构建输入到神经网络。在后面的章节中，我们
目前几个这样的例子。
目标编码
正如指出的
“
监督学习范式”，目标变量的确切性质可
依赖于NLP的任务得到解决。例如，在机器翻译的情况下，总结，
和问题回答，目标也文本和使用的方法，如先前被编码
onehot编码说明。
许多NLP任务实际使用分类标签，其中模型必须预测固定组中的一个
标签。编码一个常见的方法是使用每个标签的唯一索引，但这种简单的表现形式
当输出标签的数量实在太大可能成为问题。这样的一个例子是
语言建模问题，其中任务是预测下一个字，鉴于看到的话
过去。标签空间是一种语言的词汇全部，它可以很容易地增长到数
十万，其中包括特殊字符，名称等。我们在以后的重新审视这个问题
章节，看看如何解决它。
一些NLP问题涉及预测从给定文本的数值。例如，给定一个
英语作文，我们可能需要将一个数字等级或分数可读性。鉴于餐厅
回顾片段中，我们可能需要预测数值星级到第一位小数。考虑到用户的
鸣叫，我们可能需要预测用户的年龄组。有几种方法存在编码
数值目标，但简单地把目标纳入分类“仓”  - 例如，“018”，“19
25“，‘2530’，等等，并且将其视为一个序分类问题是一种合理的方法。
像素合并可以是均匀的或不均匀的和数据驱动。虽然这是一个详细的讨论
超出了本书的范围，我们提请您注意这些问题，因为目标编码
显着影响性能在这种情况下，我们鼓励你看到多尔蒂等。 （1995年）
和其中的参考文献。
计算图


总结了监督学习（训练）模式作为数据流架构，其中
输入是由模型（数学表达式）获得的预测，和损耗转化
函数（另一种表达），以提供一个反馈信号来调节该模型的参数。这个
数据流可以使用计算的图形数据结构来方便地实现。
从技术上讲，计算图形是一个抽象模型的数学表达式。在里面
深学习的情况下，计算图的实现（例如Theano，
TensorFlow和PyTorch）做额外的簿记来实现所需的自动分化
在监督学习范式训练过程中获得的参数梯度。我们探讨这个
进一步
“
PyTorch基础”。推断（或预测）是简单地表达评价（前向气流
上的计算图）。让我们来看看如何计算图模型表达。考虑
表达：
这可以写成两个子表达式，Z = WX和Y = Z + B。然后，我们可以代表原
表达使用有向无环图（DAG），其中节点是数学运算，
比如乘法和加法。到的操作的输入是输入边缘节点和
各操作的输出是出射边缘。因此，对于表达式y = WX + B，计算
如所示曲线图
F
igure 16。在下面的章节中，我们看到PyTorch如何使我们能够创建
在一个简单的方式计算图表，以及它如何使我们能够计算出梯度
没有与任何有关簿记自己。
图16。代表Y = WX + B使用计算曲线图。
PyTorch基础知识
在这本书中，我们广泛使用PyTorch实现我们的深度学习模式。 PyTorch是
开源，communitydriven深学习框架。不像Theano，来自Caffe和TensorFlow，
PyTorch实现tapebased
一个
utomatic分化方法，使我们能够定义和
动态执行的计算图表。这是用于调试和也极有帮助
构建复杂的模型以最小的努力。


YAMMIC与静态计算图
像Theano，来自Caffe和TensorFlow静态框架要求的计算图表是第一
声明，编译，然后执行。虽然这会导致效率极高
实现（在生产和移动设置有用），它可以成为相当麻烦
研发过程中。现代的框架，如Chainer，DyNet和PyTorch
实行动态的计算图表，允许更灵活，风格势在必行
开发，而不需要每次执行之前编译模型。动态
计算图是特别有用的建模NLP任务，每个输入可能
潜在地导致不同的图形结构。
PyTorch是一个优化的张量操作库，提供了深度学习的封装阵列。
在图书馆的核心是张量，这是一个数学对象持有一些多维
数据。零阶张量仅仅是一个数字，或者一个标量。一阶（1storder张量）的张量是一个
号的阵列或向量。类似地，2ndorder张量是向量的阵列，或矩阵。
因此，张量可以概括为标量的n维阵列中，如在示出的
F
igure 17。
图17。张量作为多维数组的概括。
在本节中，我们把我们的第一个步骤，PyTorch各种PyTorch熟悉你
操作。这些包括：
创建张量
与张量操作
索引，切片，并用张量加盟
计算与张量梯度
使用CUDA张量与图形处理器
我们建议，在这一点上，你有一个Python 3.5+笔记本愿与PyTorch安装，如
接下来的描述，和你的例子跟随。我们还建议通过工作
练习在本章后面。
安装PyTorch
第一步是在您的计算机上安装PyTorch通过选择系统首选项
p
ytorch.org。选择您的操作系统，然后包管理器（建议康达或
PIP），随后的Python的版本，您正在使用（推荐3.5+）。这将产生
你要执行的命令来安装PyTorch。截至发稿时，install命令
康达环境，例如，如下：

```sh
conda install pytorch torchvision -c pytorch
```
注意
如果你有一个CUDA enabled图形处理器单元（GPU），你也应该选择
相应的版本CUDA的。有关详细信息，请按照安装说明
pytorch.org。



创建张量
首先，我们定义一个辅助函数，描述（X），这将总结了张量的各种属性
的x，如张量，该张量的尺寸，和张量的内容的类型：
输入[0] def描述（x）：
打印（ “类型：{}”。格式（x.type（）））
打印（ “形状/尺寸：{}”。格式（x.shape））
打印（ “值：\ N {}”。格式（X））
PyTorch允许我们创建使用的火炬包许多不同的方式张量。一种方法
创建一个张量是通过指定它的尺寸来初始化一个随机，如图
Ë
xample 13。
实施例13。创建于PyTorch与torch.Tensor张量
输入[0]导入火炬
描述（torch.Tensor（2，3））
输出[0]类型：torch.FloatTensor
形状/尺寸：torch.Size（[2,3]）
价值观：
张量（[[3.2018e05，4.5747e41，2.5058e + 25]，
[3.0813e41，4.4842e44，0.0000E + 00]]）
我们还可以通过随机与值从上均匀分布初始化它创建一个张量
interval [0,1]或标准正态分布如图所示
Ë
xample 14。随机
从统一分布来看，初始化的张量非常重要，正如您将在第3章和第3章中看到的那样
4。
实施例14。创建一个随机初始化张量


- input 
```python
import torch
describe(torch.rand(2, 3))   # uniform random
describe(torch.randn(2, 3))  # random normal
```
- output 
```
Type:  torch.FloatTensor
Shape/size:  torch.Size([2, 3])
Values:
 tensor([[ 0.0242,  0.6630,  0.9787],
        [ 0.1037,   0.3920,  0.6084]])
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[-0.1330, -2.9222, -1.3649],
       [  2.3648,  1.1561,  1.5042]])
```
我们也可以创建张量都充满了同样的标量。对于创建的0或1张量，我们
有内置函数，以及用于与特定值填充它，我们可以使用fill_（）方法。 任何
PyTorch方法用下划线（_）指的是就地操作;即，它修改
代替内容而不创建新对象，如图
实施例15。
- input
```python
import torch
describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
describe(x)
x.fill_(5)
describe(x)
```
- output
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 5.,  5.,  5.],
       [ 5.,  5.,  5.]])
```
例如16演示如何，我们也可以通过使用Python列表声明创建一个张量。

实施例16。创建并从列表中初始化张量


- input
```python
x = torch.Tensor([[1, 2, 3], 
                 [4, 5, 6]])
describe(x)
```
- output
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2., 3.],
       [ 4.,  5., 6.]])
```
值既可以来自一个列表，如在前面的例子中，或从一个NumPy的阵列。而且，中
当然，我们总是可以从PyTorch张去NumPy的阵列，以及。注意的类型
张量是DoubleTensor而不是默认的FloatTensor（见下节）。 这个
与NumPy的随机矩阵，float64的数据类型对应，如在呈现
实施例1-7

例1-7。创建并从NumPy的初始化张量
- input
```python
import torch
import numpy as np
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))
```
- output
```
Type: torch.DoubleTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.8360,  0.8836,  0.0545],
       [ 0.6928,  0.2333,  0.7984]], dtype=torch.float64)
```
工作时与NumPy阵列和PyTorch张量之间转换的能力就变得很重要
与使用NumPy formatted数值旧版库。
张量类型和尺寸
各张量具有相关联的类型和大小。当您使用默认的张量型
torch.Tensor构造是torch.FloatTensor。但是，你可以将张量转换成
不同类型（浮点型，长，双，等）通过在初始化指定它或稍后使用的一个
类型转换方法。有指定初始化式两种方式：要么通过直接调用
特定类型张量的构造，如FloatTensor或LongTensor，或者使用特殊的
方法，torch.tensor()，并提供dtype，如图
实施例1-8。
实施例1-8。张量特性


- input 0
```python
x = torch.FloatTensor([[1, 2, 3], 
                      [4, 5, 6]])
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])
```
- input 1
```python
x = x.long()
describe(x)
```
- output 1
```
Type: torch.LongTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1,  2,  3],
       [ 4,  5,  6]])
```
- input 2
```python
x = torch.tensor([[1, 2, 3],
                 [4, 5, 6]], dtype=torch.int64)
describe(x)
```
- output 2
```
Type: torch.LongTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1,  2,  3],
       [ 4,  5,  6]])
```
- input 3
```
x = x.float()
describe(x)
```
- output 3
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])
```
我们使用张量物体的形状属性和尺寸（）方法来访问的测量它的
尺寸。访问这些测量的两种方法大多是同义的。检查
张量的形状是在调试PyTorch代码不可缺少的工具。
张量操作
您已经创建了张量后，可以对它们进行操作，就像您用传统的做

你可以对它们进行操作，就像您用传统的做
编程语言类型，如+， - ，*，/。取而代之的是运营商的，你也可以像使用功能
.add（），如例1-9所示，对应于符号运算符。

例1-9 张量操作：加成
- input 0
```python
import torch
x = torch.randn(2, 3)
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0461,  0.4024, -1.0115],
       [ 0.2167, -0.6123,  0.5036]])
```

 - input 1
 ```python
 describe(torch.add(x, x))
 ```
- output 1
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0923,  0.8048, -2.0231],
       [ 0.4335, -1.2245,  1.0072]])
```
- input 2
```python
describe(x + x)
```

- output 2
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.0923,  0.8048, -2.0231],
       [ 0.4335, -1.2245,  1.0072]])
```

还有一些操作可以应用于张量的特定维度。正如你可能有
已经注意到，对于我们所代表的行作为维0和列维度1二维张量，如
例1-10中所示。

例1-10 基于维度的张量运算
- input 0
```pyhton
import torch
x = torch.arange(6)
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([6])
Values:
tensor([ 0.,  1.,  2.,  3.,  4.,  5.])
```
- input 1
```python
x = x.view(2, 3)
describe(x)
```

- output 1
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]])
```
- input 2
```
describe(torch.sum(x, dim=0))
```

- output 2
```
Type: torch.FloatTensor
Shape/size: torch.Size([3])
Values:
tensor([ 3.,  5.,  7.])
```

- input 3
```python
describe(torch.sum(x, dim=1))
```
- output 3
```
Type: torch.FloatTensor
Shape/size: torch.Size([2])
Values:
tensor([  3.,  12.])
```

- input 4
```python
describe(torch.transpose(x, 0, 1))
```

- output 4
```
Type: torch.FloatTensor
Shape/size: torch.Size([3, 2])
Values:
tensor([[ 0.,  3.],
       [ 1.,  4.],
       [ 2.,  5.]])
```

通常，我们需要执行更复杂的操作，包括索引，切片，连接和突变的组合。像NumPy这样和其他数字图书馆，PyTorch具有内置功能
做出这样的张量的操作非常简单。

索引，切片和连接
如果你是一个NumPy的用户，PyTorch的索引和切片方案，在所示
例1-11，可能是
非常熟悉。
例1-11。切片和索引张量

- input 0
```python
import torch
x = torch.arange(6).view(2, 3)
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]]
```
 - input 1
 ```python
 describe(x[:1, :2])
 ```
 - output 1
 ```
 Type: torch.FloatTensor
Shape/size: torch.Size([1, 2])
Values:
tensor([[ 0.,  1.]])
 ```
 - input 2
 ```python
 describe(x[0, 1])
 ```
 - output 2
 ```
 Type: torch.FloatTensor
Shape/size: torch.Size([])
Values:
1.0
 ```
 例112证明，PyTorch还具有用于复杂索引和切片功能
操作，在这里你可能会感兴趣的有效访问张量的不连续的位置。


实施例112。复合索引：张量的不连续的索引
- input 0
```python
indices = torch.LongTensor([0, 2])
describe(torch.index_select(x, dim=1, index=indices))
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[ 0.,  2.],
       [ 3.,  5.]])
```

- input 1
```python
indices = torch.LongTensor([0, 0])
describe(torch.index_select(x, dim=0, index=indices))
```
- output 1
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
       [ 0.,  1.,  2.]])
```
- input 2
```python
row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])
```
- output
```
Type: torch.FloatTensor
Shape/size: torch.Size([2])
Values:
tensor([ 0.,  4.]
```

请注意，该指数是一个LongTensor;这是利用PyTorch索引的要求
功能。我们还可以使用内置拼接功能加入张量，如图
例1-13 通过指定张量和尺寸
- input 0
```python
import torch
x = torch.arange(6).view(2,3)
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]])
```
- input 0
```
import torch
x = torch.arange(6).view(2,3)
describe(x)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
     [ 3.,  4.,  5.]])
```

- input 1
```python
describe(torch.cat([x, x], dim=0))
```
- output 1
```
Type: torch.FloatTensor
Shape/size: torch.Size([4, 3])
Values:
tensor([[ 0.,  1.,  2.],
     [ 3.,  4.,  5.],
     [ 0.,  1.,  2.],
     [ 3.,  4.,  5.]])
```

- input 2
```python
describe(torch.cat([x, x], dim=1))
```
- output 2
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 6])
Values:
tensor([[ 0.,  1.,  2.,  0.,  1.,  2.],
     [ 3.,  4.,  5.,  3.,  4.,  5.]])
```

- input 3
```
describe(torch.stack([x, x]))
```
- output 3
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2, 3])
Values:
tensor([[[ 0.,  1.,  2.],
     [ 3.,  4.,  5.]],
     [[ 0.,  1.,  2.],
      [ 3.,  4.,  5.]]])
```
PyTorch还实现上张量高效线性代数运算，如乘法，
求逆，和追踪，你可以在例1-14看到。

- input 0
```python
import torch
x1 = torch.arange(6).view(2, 3)
describe(x1)
```
- output 0
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values:
tensor([[ 0.,  1.,  2.],
     [ 3.,  4.,  5.]])
```
- input
```python
x2 = torch.ones(3, 2)
x2[:, 1] += 1
describe(x2)
```
- output 1
```
Type: torch.FloatTensor
Shape/size: torch.Size([3, 2])
Values:
tensor([[ 1.,  2.],
     [ 1.,  2.],
     [ 1.,  2.]])
```

- input 2
```python
describe(torch.mm(x1, x2))
```
- output 2
```
Type: torch.FloatTensor
Shape/size: torch.Size([2, 2])
Values:
tensor([[  3.,   6.],
     [ 12.,  24.]])
```
到目前为止，我们已经研究了创建和操作常量PyTorch张量对象的方法。 正如编程语言（例如Python）具有封装一段数据的变量并且具有关于该数据的附加信息（例如，存储它的存储器地址），PyTorch张量器处理构建计算图形所需的簿记。 机器学习只需在实例化时启用布尔标志即可。

# 张量和计算图
PyTorch张量类封装了数据（张量本身）和一系列操作，如代数运算，索引和整形操作。 但是，如示例115所示，当在张量上将requires_grad布尔标志设置为True时，启用簿记操作，该操作可以跟踪张量的梯度以及梯度函数，这两者都是为了便于在TheSupervisedLearningParadigm中讨论的梯度学习所需要的。。


例1-15 为梯度簿记创建张量
- input 0
```python
import torch
x = torch.ones(2, 2, requires_grad=True) 
describe(x)
print(x.grad is None)
```

- output 0
```
Type: torch.FloatTensor 
Shape/size: torch.Size([2, 2]) 
Values:
tensor([[ 1., 1.],
[ 1., 1.]]) True
```

- input 1
```python
y = (x + 2) * (x + 5) + 3 
describe(y)
print(x.grad is None)
```

- output 1
```
Type: torch.FloatTensor 
Shape/size: torch.Size([2, 2]) 
Values:
tensor([[ 21., 21.],
[ 21., 21.]]) True
```

- input 2
```python
z = y.mean()
describe(z) 
z.backward() 
print(x.grad is None)
```

- output 2
```
Type: torch.FloatTensor 
Shape/size: torch.Size([]) 
Values:
21.0
False
```

当您使用`requires_grad = True`创建张量时，您需要PyTorch来管理计算渐变的簿记信息。 首先，PyTorch将跟踪前向传球的值。 然后，在计算结束时，使用单个标量来计算后向传递。 通过在评估损失函数时产生的张量上使用`backward（）`方法来启动向后传递。 向后传递计算参与正向传递的张量对象的梯度值。
通常，梯度是表示函数输出相对于函数输入的斜率的值。 在计算图设置中，模型中的每个参数都存在梯度，可以将其视为参数对误差信号的贡献。 在PyTorch中，您可以使用.grad成员变量访问计算图中节点的渐变。 优化器使用`.grad`变量来更新参数的值。
CUDA Tensors
到目前为止，我们一直在CPU内存上分配我们的张量。在进行线性代数运算时，如果你有GPU，那么使用GPU可能是有意义的。要使用GPU，您需要首先在GPU的内存上分配张量。通过名为CUDA的专用API访问GPU。 CUDA API由NVIDIA创建，仅限于在NVIDIA GPU上使用.9 PyTorch提供的CUDA张量对象在使用时与常规CPU绑定器无法区分，除了它们在内部分配的方式。
PyTorch可以很容易地创建这些CUDA张量，将张量从CPU传输到GPU，同时保持其基础类型。 PyTorch中的首选方法是与设备无关，并编写无论是在GPU还是在CPU上运行的代码。在示例116中，我们首先使用torch.cuda.is_available（）检查GPU是否可用，并使用torch.device（）检索设备名称。然后，通过使用.to（设备）方法实例化所有未来的张量并将其移动到目标设备。

例1-16 创建CUDA张量

- input 0
```python
import torch
print (torch.cuda.is_available())
```

- output 0
```
True
```

- input 1
```python
# preferred method: device agnostic tensor instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print (device)
```

- output 1
```
cuda
```

- input 2
```python
x = torch.rand(3, 3).to(device) 
describe(x)
```

- output 2
```
Type: torch.cuda.FloatTensor 
Shape/size: torch.Size([3, 3]) 
Values:
tensor([[ 0.9149, 0.3993, 0.1100],
[ 0.2541, 0.4333, 0.4451],
[ 0.4966, 0.7865, 0.6604]], device='cuda:0')
```
要对CUDA和nonCUDA对象进行操作，我们需要确保它们位于同一设备上。 如果我们不这样做，计算就会中断，如例子117所示。 例如，当计算不属于计算图的监视度量时，会出现这种情况。 在两个张量对象上操作时，请确保它们都在同一设备上。
例1-17 将CUDA张量与CPU绑定张量混合

- input 0

```python
y = torch.rand(3, 3) x+y
```

- output 0
```
------------------------------------------------- 
RuntimeError Traceback (most recent call last)
1 y = torch.rand(3, 3) ---> 2 x + y
RuntimeError: Expected object of type
torch.cuda.FloatTensor but found type torch.FloatTensor for argument #3 '
```
- input 1
```python
cpu_device = torch.device("cpu") 
y = y.to(cpu_device)
x = x.to(cpu_device)
x+y
```
- output 1
```
tensor([[ 0.7159, 1.0685, 1.3509], [ 0.3912, 0.2838, 1.3202],
[ 0.2967, 0.0420, 0.6559]])
```
请记住，从GPU来回移动数据是很昂贵的。 因此，典型的过程涉及在GPU上执行许多可并行化的计算，然后将最终结果传送回CPU。 这将允许您充分利用GPU。 如果您有多个CUDAvisible设备（即多个GPU），最佳做法是在执行程序时使用CUDA_VISIBLE_DEVICES环境变量，如下所示：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

作为本书的一部分，我们不涉及并行性和multiGPU培训，但它们在扩展实验中是必不可少的，有时甚至是训练大型模型。 我们建议您参考
yTorch文档和论坛，以获得有关此主题的其他帮助和支持。
# 练习
掌握主题的最佳方法是解决问题。 这里有一些热身练习。 许多问题需要经过官方的观察并找到有用的功能。
1.创建2D张量，然后添加在尺寸0处插入的尺寸为1的尺寸。
2.删除刚添加到先前张量的额外尺寸。
3.在区间[3,7]中创建一个形状为5x3的随机张量
4.使用正态分布中的值创建张量（mean = 0，std = 1）。
5.检索张量torch.Tensor（[1,1,1,0,1]）中所有非零元素的索引。
。创建一个大小为(3,1)的随机张量，然后将四个副本水平堆叠在一起。

7.返回两个三维矩阵的批量矩阵矩阵乘积
`(a = torch.rand(3,4,5)，b = torch.rand(3,5,4))`。

8.返回3D矩阵和2D矩阵的批量矩阵矩阵乘积
`(a = torch.rand(3,4,5)，b = torch.rand(5,4))`。
解决方案

1. `a = torch.rand(3,3)a.unsqueeze(0)`

2. `a.squeeze(0)`

3. `3 + torch.rand(5,3)`

4. `a = torch.rand(3,3)`
`a.normal_()`

5. `a = torch.Tensor([1，`
  `torch.nonzero(a)`中

6. `a = torch.rand(3,1)`
  `a.expand(3,4)`

7. `a = torch.rand(3, 4, 5)`
    `b = torch.rand(3, 5, 4)`
    `torch.bmm(a, b)`
8. `a = torch.rand(3,4,5)`
    `b = torch.rand(5,4)`
    
    `torch.bmm(a，b.unsqueeze(0).expand(a.size(0)，* b.size())`
# 小结
在本章中，我们介绍了本书的主要内容 - 自然语言处理(NLP)和深度学习 - 并对监督学习范式进行了详细的理解。您现在应该熟悉或至少知道各种相关术语，例如观察，目标，模型，参数，预测，损失函数，表示，学习/训练和推理。您还了解了如何使用onehot编码对学习任务的输入(观察和目标)进行编码，我们还检查了基于计数的表示，如TF和TFIDF。我们首先探索计算图是什么，然后考虑静态与动态计算图并参观PyTorch的张量操纵操作，开始了我们的PyTorch之旅。在第2章中，我们提供了传统NLP的概述。如果您对本书的主题不熟悉并为其他章节做好准备，这两章应该为您奠定必要的基础。
例1-17 将CUDA张量与CPU绑定张量混合

- input 0

```python
y = torch.rand(3, 3) x+y
```

- output 0
```
------------------------------------------------- 
RuntimeError Traceback (most recent call last)
1 y = torch.rand(3, 3) ---> 2 x + y
RuntimeError: Expected object of type
torch.cuda.FloatTensor but found type torch.FloatTensor for argument #3 '
```
- input 1
```python
cpu_device = torch.device("cpu") 
y = y.to(cpu_device)
x = x.to(cpu_device)
x+y
```
- output 1
```
tensor([[ 0.7159, 1.0685, 1.3509], [ 0.3912, 0.2838, 1.3202],
[ 0.2967, 0.0420, 0.6559]])
```
请记住，从GPU来回移动数据是很昂贵的。 因此，典型的过程涉及在GPU上执行许多可并行化的计算，然后将最终结果传送回CPU。 这将允许您充分利用GPU。 如果您有多个CUDAvisible设备（即多个GPU），最佳做法是在执行程序时使用CUDA_VISIBLE_DEVICES环境变量，如下所示：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

作为本书的一部分，我们不涉及并行性和multiGPU培训，但它们在扩展实验中是必不可少的，有时甚至是训练大型模型。 我们建议您参考
yTorch文档和论坛，以获得有关此主题的其他帮助和支持。
# 练习
掌握主题的最佳方法是解决问题。 这里有一些热身练习。 许多问题需要经过官方的观察并找到有用的功能。
1.创建2D张量，然后添加在尺寸0处插入的尺寸为1的尺寸。
2.删除刚添加到先前张量的额外尺寸。
3.在区间[3,7]中创建一个形状为5x3的随机张量
4.使用正态分布中的值创建张量（mean = 0，std = 1）。
5.检索张量torch.Tensor（[1,1,1,0,1]）中所有非零元素的索引。
# 参考

1. PyTorch官方API文档。
2. Dougherty, James, Ron Kohavi, and Mehran Sahami. (1995). “Supervised and Unsupervised Discretization of Continuous Features.” Proceedings of the 12th International Conference on Machine Learning.
3. Collobert, Ronan, and Jason Weston. (2008). “A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning.” Proceedings of the 25th International Conference on Machine Learning.

<hr>

1. 虽然神经网络和NLP的历史悠久而丰富，但Collobert和Weston（2008）经常被认为是对NLP采用现代风格应用深度学习的先驱。

2. 分类变量是一个采用一组固定值的变量;例如，{TRUE，FALSE}，{VERB，NOUN，ADJECTIVE，...}，以及更多其他的。

3. 深度学习与2006年之前的文献中讨论的传统神经网络不同，它指的是越来越多的技术，通过添加更多的网络工作.在章节3和4中我们会研究为什么这很重要。

4. “序数”分类是多类分类问题，其中标签之间存在部分顺序。在我们的年龄示例中，类别“0-18”出现在“19-25”之前，依此类推。

5. Seppo Linnainmaa首先在计算图上引入了自动微分的想法，作为他1970年硕士论文的一部分！其中的变体成为现代深度学习框架的基础，如Theano，TensorFlow和PyTorch。

6. 从v1.7开始，TensorFlow有一个“急切模式（eager mode）”，它使得在执行之前无需编译图形，但静态图仍然是TensorFlow的支柱。

7. 您可以在本书的gitHub repo中的 */chapters/chapter_1/PyTorch_Basics.ipynb* 下找到此部分的代码。

8. 标准正态分布是正态分布，均值`= 0`且方差`= 1`，

9. 这意味着如果你有一个非NVIDIA GPU，比如说AMD或ARM，那么你在写这篇文章时就不走运了（当然，你仍然可以在CPU模式下使用PyTorch）。但是，他将来可能会改变。

