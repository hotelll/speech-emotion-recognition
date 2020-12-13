# 语音情绪识别

本项目的目标是设计一个情绪识别系统，要求在短语音（1s 左右）情况下，能够区分**快乐**，**悲伤**，**愤怒**和**中性**这四种区分度大的情绪。

本项目以下项目为参考：

> https://github.com/Renovamen/Speech-Emotion-Recognition

## 项目概览

### 语音特征提取

本项目采用 openSMILE 的 IS10_paraling 特征集进行特征提取。

### 数据集

本项目采用 RAVDESS 英文数据集，这是一个表演型数据集，包含 12 男 12 女七种情感的 wav 文件，本人做了以下处理来满足要求：

- 将语音切成 1s 长度，共享标签；
- 删除了平静、惊讶、厌恶三个标签的数据。

数据集在 dataset 文件夹中。

### 训练

本项目在比较了 LSTM，CNN，SVM 和 MLP 后，选择采用效果最好的 CNN 进行训练。

## 环境

- scikit-learn
- keras
- tensorflow
- pandas
- matplotlib
- numpy
- opensmile

## 使用

### 配置环境

```shell
pip install -r requirement.txt
```

安装 [Opensmile](https://github.com/naxingyu/opensmile)

### 预测 Demo

```shell
python predict.py
```

直接运行 `predict.py`，使用已经预先训练的模型预测你输入语音（wav文件）的情感。可以通过修改 `predict.py` 中的 `audio_path` 来修改要识别的语音。在项目路径 `test_speech/` 中给了一些可以用来测试的语音，其均来自于 TESS 数据集，与训练集不重合。预测结果展示为雷达图，如下为我们输入一段愤怒语音的输出结果。

<img src="README\radar.png" alt="radar" style="zoom:50%;" />

### 预处理

首先我们利用 openSMILE 进行数据预处理，提取 RAVDESS 的特征保存到路径 `features/train_cnn1d_opensmile_ravdess_IS10.csv` 文件中。

```shell
python preprocess.py
```

### 训练

运行 `train.py` 进行模型训练。具体训练参数设置在 `cnn1d.yaml`。

```shell
python train.py
```

训练可以得到准确率和损失的曲线图。

<center> <img src="README\image-20201213184107148.png" style="zoom:45%;"/><img src="README\image-20201213184128736.png" style="zoom:45%;"/> </center>

目前我们项目的准确率在76%左右。

训练的模型保存在 `checkpoints/`文件夹下。

### 可修改处

- 新的模型可以定义在`/models`文件夹
- 同时可以选用 openSMILE 不同的特征集（在 cnn1d.yaml 中修改）
- 可以选择新的数据集放置在`dataset/`下，注意一种按照情感分成多个文件夹，与原来的格式保存一致，情感种类可以增加
- 可以在 yaml 文件下修改超参

