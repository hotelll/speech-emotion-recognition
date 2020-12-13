from .dnn import CNN1D

'''
setup(): 创建模型

输入:
    config(Class)
    n_feats(int): 特征数量（神经网络输入张量大小）
'''
def setup(config, n_feats):
    model = CNN1D(
        input_shape = n_feats,
        num_classes = len(config.class_labels),
        lr = config.lr,
        n_kernels = config.n_kernels,
        kernel_sizes = config.kernel_sizes,
        hidden_size = config.hidden_size,
        dropout = config.dropout
    )

    return model
