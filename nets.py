import paddle.fluid as fluid


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    Bow net

    Bow(Bag Of Words)模型，是一个非序列模型。使用基本的全连接结构
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    auc_out = fluid.layers.auc(input=prediction, label=label)
    return avg_cost, acc, auc_out[1], prediction


def cnn_net(data,
            label,
            dict_dim,
            emb_dim=256,
            hid_dim=256,
            hid_dim2=96,
            class_dim=2,
            win_size=3):
    """
    Conv net

    浅层CNN模型，是一个基础的序列模型，能够处理变长的序列输入，提取一个局部区域之内的特征
    """

    # embedding layer
    # one-hot to word-embedding
    # 嵌入层 该层用于查找由输入提供的id在查找表中的嵌入矩阵。查找的结果是input里每个ID对应的嵌入矩阵。 所有的输入变量都作为局部变量传入LayerHelper构造器
    # 词向量层: 将词语转化为固定维度的向量，利用向量之间的距离来表示词之间的语义相关程度。将得到的词向量定义为行向量，再将语料中所有的单词产生的行向量拼接在一起组成矩阵
    # size(tuple | list) - 查找表参数的维度。应当有两个参数，一个代表嵌入矩阵字典的大小，一个代表每个嵌入向量的大小。
    # is_sparse(bool) - 代表是否用稀疏更新的标志

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # convolution layer *
    # sequence_conv_pool由序列卷积和池化组成
    # input (Variable) - sequence_conv的输入，支持变量时间长度输入序列。当前输入为shape为（T，N）的矩阵，T是mini-batch中的总时间步数，N是input_hidden_size
    # num_filters （int）- 滤波器数 The number of filter. It is as same as the output feature channel.
    # filter_size （int）- 滤波器大小
    # act （str） - Sequence_conv层的激活函数类型。默认：sigmoid
    #   act select：一般用relu， 若使用sigmoid/tanh作为激活函数则要对input进行归一化，否则激活后的值会进入平坦区，使隐层输出全部相同
    #   sigmoid 1/(1+e^(-x)) 饱和时梯度值非常小，输出值不是以0为中心；tanh 2σ(2x)-1 仍具有饱和问题
    #   relu max(0,x) 很大程度上解决了BP算法在优化深层神经网络时的梯度耗散问题，x>0无梯度耗散且收敛快，x<0增大稀疏性且泛化性能好，运算量很小；可能造成很多永远处于die的神经元，需要tradeoff，需注意学习率的设置和死亡节点所占比例
    # pool_type （str）- 池化类型。可以是max-pooling的max，average-pooling的average，sum-pooling的sum，sqrt-pooling的sqrt。默认max
    #   池化往往在卷积后面，通过池化降低卷积层输出的特征向量，同时改善结果（不容易出现过拟合）
    # 返回：序列卷积（Sequence Convolution）和池化（Pooling）的结果
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="relu",  # original_act: tanh
        pool_type="max")

    # full connect layer
    # 全连接层 可同时将多个tensor（ input 可使用多个tensor组成的一个list）作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。
    # FC层用每个tensor和它对应的权相乘得到输出tensor。如果有多个输入tensor，那么多个乘法运算将会加在一起得出最终结果。如果 bias_attr 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 act 非空，它也会加入最终输出的计算中。
    # 起到了分类器的作用，将学到的分布式表示特征映射到样本标记空间，但参数冗余，可由convolution layer实现：前层也是是全连接的全连接层，可转为卷积核为1*1的全局卷积；前层为卷积层的全连接层，可转为卷积核为h*w的全局卷积
    # input(Variable | list of Variable) – 该层的输入tensor(s)（张量），其维度至少是2
    # size(int) – 该层输出单元的数目
    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim)
    # dropout 防止过拟合 提升模型泛化能力
    fc_1_drop = fluid.layers.dropout(fc_1, dropout_prob=0.5)

    # softmax layer
    # 二分类问题可用softmax作为最后的激活层；多分类单标签问题可用sigmoid作为最后的激活层，取概率最高的作为结果；多标签问题可用多个softmax作为最后的激活层，相当于把每一个类别当作二分类来处理
    # softmax函数的本质就是将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间并且和为1。
    # sigmoid把一个值映射到0-1之间。
    prediction = fluid.layers.fc(input=[fc_1_drop], size=class_dim, act="softmax")
    # cross_entropy
    # 支持standard cross-entropy computation（标准交叉熵损失计算） 以及soft-label cross-entropy computation（软标签交叉熵损失计算）
    # input (Variable|list) – 一个形为[N x D]的二维tensor，其中N是batch大小，D是类别（class）数目。 这是由之前的operator计算出的概率，绝大多数情况下是由softmax operator得出的结果
    # label (Variable|list) – 一个二维tensor组成的正确标记的数据集(ground truth)。 当 soft_label 为False时(default)，label为形为[N x 1]的tensor<int64>。 soft_label 为True时, label是形为 [N x D]的 tensor<float/double>
    # 返回： 一个形为[Nx1]的二维tensor，承载了交叉熵损失
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    # Area Under the Curve(AUC) Layer
    # 该层根据前向输出和标签计算AUC，在二分类(binary classification)估计中广泛使用 ROC:
    # curve(str)曲线类型，可以为 ROC 或 PR，默认 ROC。auc curve(str): Curve type, can be 'ROC' or 'PR'. Default 'ROC'.
    # 返回：代表当前AUC的scalar auc_out = [auc_out, batch_auc_out,...]
    # ROC: 受试者工作特征曲线(receiver operating characteristic),roc曲线上每个点反映着对同一信号刺激的感受性。
    #   真负类率(True Negative Rate)TNR: TN/(FP+TN), TNR=1-FPR。Specificity
    #   横轴：负正类率(false postive rate FPR)特异度，划分实例中所有负例占所有负例的比例；(1-Specificity) 1-TNR 负正类率(False Postive Rate)FPR: FP/(FP+TN)
    #   纵轴：真正类率(true postive rate TPR)灵敏度，Sensitivity 真正类率 TPR: TP/(TP+FN)
    #   横轴FPR越大，预测正类中实际负类越多。纵轴TPR越大，预测正类中实际正类越多。
    #   理想目标：TPR=1，FPR=0,即图中(0,1)点，故ROC曲线越靠拢(0,1)点，越偏离45度对角线越好，Sensitivity、Specificity越大效果越好。
    #   https://www.cnblogs.com/dlml/p/4403482.html
    # PR: 准确率召回率曲线
    auc_out_list = fluid.layers.auc(input=prediction, label=label)
    batch_auc_out = auc_out_list[1]
    return avg_cost, acc, batch_auc_out, prediction


def lstm_net(data,
             label,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0):
    """
    Lstm net

    单层LSTM模型，序列模型，能够较好地解决序列文本中长距离依赖的问题
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    # Lstm layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    # max pooling layer
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)

    # full connect layer
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    auc_out = fluid.layers.auc(input=prediction, label=label)
    return avg_cost, acc, auc_out[1], prediction


def bilstm_net(data,
               label,
               dict_dim,
               emb_dim=128,
               hid_dim=128,
               hid_dim2=96,
               class_dim=2,
               emb_lr=30.0):
    """
    Bi-Lstm net

    双向单层LSTM模型，序列模型，通过采用双向lstm结构，更好地捕获句子中的语义特征。AI平台上情感倾向分析模块采用此模型进行训练和预测
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    # bi-lstm layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    rfc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0, size=hid_dim * 4, is_reverse=True)

    # extract last layer
    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)

    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    # concat layer
    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)

    # full connect layer
    fc1 = fluid.layers.fc(input=lstm_concat, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    auc_out = fluid.layers.auc(input=prediction, label=label)
    return avg_cost, acc, auc_out[1], prediction


def gru_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=30.0):
    """
    gru net

    单层GRU模型，序列模型，能够较好地解序列文本中长距离依赖的问题
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)

    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)

    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)

    fc1 = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='relu')

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    auc_out = fluid.layers.auc(input=prediction, label=label)
    return avg_cost, acc, auc_out[1], prediction
