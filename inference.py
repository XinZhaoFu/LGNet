# coding=utf-8
from data_utils.utils import get_specific_type_file_list, test_img_init, inference_check, get_inference_img
import cv2
import os
from model.lgnet import LGNet
from model.unetm import UNetM
import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
from data_utils.data_loader import Data_Loader_File


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(threshold=np.inf)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def seg_predict(checkpoint_save_path, test_file_path, predict_save_path):
    """

    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型

    # model = LGNet()
    model = UNetM()
    model.load_weights(checkpoint_save_path)

    test_file_path_list = get_specific_type_file_list(test_file_path, 'jpg')
    test_file_path_list = [test_file_path_list[-1]]

    for test_file in tqdm(test_file_path_list):

        test_img, test_info = test_img_init(test_file)

        predict_temp = model.predict(test_img)
        # print(predict_temp)

        predict_temp = tf.math.argmax(predict_temp, 3)
        predict_temp = np.array(predict_temp)
        predict_temp = np.reshape(predict_temp, newshape=(512, 512))
        # print(predict_temp)
        predict_img = np.zeros(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict_temp[:, :] * 127

        # cv2.imwrite(predict_save_path + (test_file.split('/')[-1]).split('.')[0] + '.png', predict_img)


def seg_predict_batch(checkpoint_save_path, test_file_path, predict_save_path):
    """

    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """
    print('[info]模型加载 图片加载')
    # 加载模型

    model = LGNet()
    model.load_weights(checkpoint_save_path)

    test_file_path_list = get_specific_type_file_list(test_file_path, 'jpg')
    # test_file_path_list = [test_file_path_list[-1]]

    for test_file in tqdm(test_file_path_list):
        test_img = cv2.imread(test_file)
        test_img = test_img / 255.
        # test_img = np.array(test_img, dtype=np.float) / 255.
        test_img = np.reshape(test_img, newshape=(1, 512, 512, 3))

        predict_temp = model.predict(test_img)
        # print(predict_temp)

        predict_temp = tf.math.argmax(predict_temp, 3)
        predict_temp = np.array(predict_temp)
        predict_temp = np.reshape(predict_temp, newshape=(512, 512))
        # print(predict_temp)
        predict_img = np.zeros(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict_temp[:, :] * 127

        cv2.imwrite(predict_save_path + (test_file.split('/')[-1]).split('.')[0] + '.png', predict_img)


def seg_predict_by_datasets(checkpoint_save_path, predict_save_path):
    """

    :param checkpoint_save_path:
    :param predict_save_path:
    :return:
    """

    print('[info]模型加载 图片加载')
    # 加载模型

    model = LGNet()
    model.load_weights(checkpoint_save_path)
    data_loader = Data_Loader_File()
    test_datasets, name_list = data_loader.load_test_data()
    test_predict = model.predict(test_datasets)

    for index, name in enumerate(name_list):
        predict = test_predict[index, :, :, :]
        predict = tf.math.argmax(predict, 3)
        predict = np.array(predict)
        predict = np.reshape(predict, newshape=(512, 512))
        # print(predict_temp)
        predict_img = np.zeros(shape=(512, 512), dtype=np.uint8)
        predict_img[:, :] = predict[:, :] * 127

        cv2.imwrite(predict_save_path + name + '.png', predict_img)


def predict(checkpoint_save_path, test_file_path, predict_save_path):
    """

    :param checkpoint_save_path:
    :param test_file_path:
    :param predict_save_path:
    :return:
    """

    print('[info]模型加载 图片加载')
    model = LGNet()
    model.load_weights(checkpoint_save_path)
    test_file_path_list = get_specific_type_file_list(test_file_path, 'jpg')
    # test_file_path_list = [test_file_path_list[-1]]
    model = inference_check(model, checkpoint_save_path)

    for test_file in tqdm(test_file_path_list):
        predict_img = get_inference_img(model, test_file)

        cv2.imwrite(predict_save_path + (test_file.split('/')[-1]).split('.')[0] + '.png', predict_img)


def main():
    checkpoint_save_path = './checkpoint/lgnet_refuge_164946558224842/lgnet_refuge_164946558224842.ckpt'

    test_file_path = './datasets/refuge_datasets/test/img/'
    predict_save_path = './datasets/refuge_datasets/test/predict/'

    start_time = datetime.datetime.now()

    # seg_predict(checkpoint_save_path, test_file_path, predict_save_path)
    # seg_predict_by_datasets(checkpoint_save_path, predict_save_path)
    # seg_predict_batch(checkpoint_save_path, test_file_path, predict_save_path)
    predict(checkpoint_save_path, test_file_path, predict_save_path)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    main()
