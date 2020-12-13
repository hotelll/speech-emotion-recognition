import os
import numpy as np
from keras.utils import np_utils
import models
import opensmile as of
import utils.opts as opts

'''
train(): 训练模型
输入: config(Class)
输出: model: 训练好的模型
'''


def train(config):
    # 加载被 preprocess.py 预处理好的特征
    x_train, x_test, y_train, y_test = of.load_feature(config, config.train_feature_path, train=True)

    # 搭建模型
    model = models.setup(config=config, n_feats=x_train.shape[1])

    # 训练模型
    print('----- start training', config.model, '-----')
    y_train, y_val = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test) # 独热编码
    model.train(
        x_train, y_train,
        x_test, y_val,
        batch_size=config.batch_size,
        n_epochs=config.epochs
    )
    print('----- end training ', config.model, ' -----')

    # 验证模型
    model.evaluate(x_test, y_test)
    # 保存训练好的模型
    model.save_model(config)


if __name__ == '__main__':
    config = opts.parse_opt()
    train(config)
