模型评估与预测
要了解模型的性能，您应该使用定量和定性方法分析模型。定量地，测量保持测试数据上的误差确定分类器是否可以概括为看不见的示例。定性地，您可以通过查看分类器的前k个预测来为新示例开发直观的模型学习内容。
评估测试数据集
为了评估测试数据上的SurnameClassifier，我们执行与restaurantreviewtextclassificationexamplein Evaluation，Inference和Inspect“相同的例程”：wesetthe split to iterate'test'数据，调用classifier.eval（）方法，并迭代测试数据与我们对其他数据的处理方式相同。在此示例中，调用classifier.eval（）可防止PyTorch在使用测试/评估数据时更新模型参数。
该模型在测试数据上实现了约50％的准确度。如果您在随附的笔记本中运行培训例程，您会注意到培训数据的性能更高。这是因为模型总是更适合它所训练的数据，因此训练数据的性能并不表示新数据的性能。如果您跟随代码，我们建议您尝试不同大小的隐藏维度。您应该注意到性能的提高。 6但是，这种增长并不会很大（特别是与modelfrom相比：例如：SortSurnamesbyUsingaCNN“）。主要的反对意见认为折叠的单拍矢量化方法是弱表示。虽然它确实将每个姓氏紧凑地表示为单个向量，但它会丢弃字符之间的订单信息，这对于识别来源至关重要。
分类一个新的SURNAME
xample 411显示了对新姓氏进行分类的代码。给定姓氏作为字符串，该函数将首先应用矢量化过程，然后获得模型预测。请注意，我们包含apply_softmax标志，以便结果包含概率。在多项式情况下，模型预测是类概率列表。我们使用PyTorch张量max（）函数来获得最佳类，由最高预测概率表示。
                                                                    
optimizer.zero_grad（）
#step 2.计算输出
y_pred =分类器（batch_dict ['x_surname']）
#step 3.计算损失
loss = loss_func（y_pred，batch_dict ['y_nationality']）loss_batch = loss.to（“cpu”）。item（）
running_loss + =（loss_batch running_loss）/（batch_index + 1）
＃step 4.使用loss来产生gradients loss.backward（）
＃step 5.使用优化器进行渐变步骤optimizer.step（）
1“ E
 例411。使用现有模型（分类器）的推断：预测给定名称的国籍
   def predict_nationality（name，classifier，vectorizer）：vectorized_name = vectorizer.vectorize（name）vectorized_name = torch.tensor（vectorized_name）.view（1,1）result = classifier（vectorized_name，apply_softmax = True）
probability_values，indices = result.max（dim = 1）index = indices.item（）
predict_nationality = vectorizer.nationality_vocab.lookup_index（index）probability_value = probability_values.item（）
return {'nationality'：predict_nationality，'probability'：probability_value}
 回顾一个新的SURNAME的前K个预测
查看不仅仅是最佳预测通常很有用。例如，NLP中的标准做法是采用前k个最佳预测并使用其他模型重新排列它们。 PyTorch提供了一个torch.topk（）函数，它提供了一种方便的方法来获得这些预测，如xample 412所示。
实施例412。预测topk国籍
                 def predict_topk_nationality（name，classifier，vectorizer，k = 5）：vectorized_name = vectorizer.vectorize（name）
vectorized_name = torch.tensor（vectorized_name）.view（1,1）prediction_vector = classifier（vectorized_name，apply_softmax = True）probability_values，indices = torch.topk（prediction_vector，k = k）
#return size是1，k
probability_values = probability_values.detach（）。numpy（）[0] indices = indices.detach（）。numpy（）[0]
results = []
对于prob_value，zip中的索引（probability_values，indices）：
nationality = vectorizer.nationality_vocab.lookup_index（index）results.append（{'nationality'：nationality，
返回结果
'概率'：prob_value}）
 规范MLP：权重正则化和结构正规化（或辍学）
在第3章中，我们解释了正则化如何成为过度拟合问题的解决方案，并研究了两种重要的权重正则化类型-L1和L2。这些权重正则化方法也适用于MLP以及卷积神经网络，我们将在下一节中讨论。除了权重正则化之外，对于深度模型（即具有多个层的模型），例如本章中讨论的前馈网络，结构...

图91。实施对话系统（使用Apple的Siri）。注意系统如何维护上下文以回答后续问题;也就是说，它知道将“他们”映射到巴拉克奥巴马的女儿身上。
演讲
话语涉及理解文本文档的整体性质。例如，话语解析的任务涉及理解两个句子在上下文中如何彼此相关。能够提供一些来自宾夕法尼亚话语树库（PDTB）的例子来说明这项任务。
             Ť
 表91。 CoNLL 2015浅层话语处理任务的示例
   话语关系的例子
    通用汽车官员希望获得他们的战略，以减少Temporal.Asynchronous.Precedence能力和在此之前的工作人员
谈判开始。
      但是那鬼不会满足于言语，他想要应变。因为。结果钱和人 - 很多。所以卡特先生成立了
三个新的陆军师并给他们一个新的
坦帕的官僚机构称为Rapid
部署力量。
     阿拉伯人只有石油。隐含=虽然这些比较。对比农民可能掌握世界的心脏
理解话语还涉及解决其他问题，如回指解析和转喻检测。在回指分辨率中，我们希望将代词的出现解析为它们所引用的实体。这可能成为一个复杂的问题，如图92所示
图92。回指解决的一些问题。在例子（a）中，“它”是指狗还是骨头？在示例（b）中，“它”既不是指它们也不是指它们。在实施例（c）和（d）中，“It”分别指玻璃和啤酒。了解啤酒更有可能起泡作用，对于解决此类指称（选择偏好）至关重要。
对象也可以是转喻，如下例所示：
北京对中国商品征收关税征收贸易关税。
在这里，北京不是指一个地方，而是指中国政府。有时，成功解决所指对象可能需要使用知识库。
                 F

 信息提取与文本挖掘
该行业遇到的常见问题类别之一涉及信息提取。我们如何从文本中提取实体（人名，产品名称等），事件和关系？我们如何将文本中的实体提及映射到知识库中的条目（又称实体发现，实体链接，填充）？3我们如何首先构建和维护该知识库（知识库群体）？这些是在不同背景下的信息提取研究中常规回答的一些问题。
文档分析和检索
另一种常见的行业NLP问题包括理解大量文档。我们如何从文档中提取主题（主题建模）？我们如何更智能地索引和搜索文档？我们如何理解搜索查询（查询解析）？我们如何为大型集合生成摘要？
NLP技术的范围和适用性很广，事实上，NLP技术可以应用于存在非结构化或半结构化数据的任何地方。作为一个例子，我们将您介绍给Dill等人。 （2007），他们应用自然语言解析技术来解释蛋白质折叠。
NLP的前沿
当该领域正在进行快速创新时，写一篇名为“NLP前沿”的部分似乎是一件愚蠢的事。但是，我们想让您一睹2018年秋季的最新趋势：
将经典的NLP文献引入可微学习范式
NLP领域已有几十年的历史，尽管深度学习领域只有几年的历史。许多创新似乎都在研究新的深度学习（可微学习）范式下的传统方法和任务。阅读经典NLP论文（我们建议阅读它们）时，一个很好的问题就是作者正在努力学习的内容。什么是输入/输出表示？如何使用前面章节中学到的技巧简化这一过程？
模型的组合性
在本书中，我们讨论了NLP的各种深度学习架构：MLP，CNN，序列模型，序列序列模型和基于注意力的模型。值得注意的是，尽管我们单独讨论了这些模型，但事实并非如此
一世

 仅仅因为教学原因。在文献中看到的一个趋势是组合不同的架构来完成工作。例如，您可以在单词的字符上编写卷积网络，然后在该表示上写入LSTM，并通过MLP完成LSTM编码的最终分类。能够根据任务需求组合构建不同的架构是深度学习中最有力的思想之一，有助于使其成功。
序列的卷积
我们在序列建模中看到的最近趋势是对模型进行建模