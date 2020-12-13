import numpy as np
from utils.common import load_model, Radar, play_audio
import opensmile as of
import utils.opts as opts


def reshape_input(data):
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data


'''
predict(): 预测音频情感

输入:
    config(Class)
    audio_path: 要预测的音频路径
	model: 加载的模型

输出: 预测结果和置信概率
'''


def predict(config, audio_path, model):
    play_audio(audio_path)
    of.get_data(config, audio_path, config.predict_feature_path, train=False)
    test_feature = of.load_feature(config, config.predict_feature_path, train=False)

    test_feature = reshape_input(test_feature)

    result = model.predict(test_feature)
    result = np.argmax(result)

    result_prob = model.predict(test_feature)[0]

    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob, config.class_labels)


if __name__ == '__main__':
    audio_path = 'test_speech/angry/1.wav'

    config = opts.parse_opt()

    # 加载模型
    model = load_model(
        checkpoint_path=config.checkpoint_path,
        checkpoint_name=config.checkpoint_name,
        model_name=config.model
    )

    predict(config, audio_path, model)
