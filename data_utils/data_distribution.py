from glob import glob
import cv2
from shutil import copyfile
from os import remove
from config.config_reader import ConfigReader
from tqdm import tqdm
from data_utils.utils import shuffle_file, recreate_dir_list, get_specific_type_file_list
import numpy as np
from loguru import logger


def label_bmp_to_png(label_path):
    """
    将label由bmp格式转为png格式

    :param label_path:
    :return:
    """
    label_file_list = glob(label_path + '*.bmp')
    print('[INFO]载入bmp格式label 共计：' + str(len(label_file_list)))
    for label_file in tqdm(label_file_list):
        label = cv2.imread(label_file)
        cv2.imwrite(label_file[:-3] + 'png', label)
        remove(label_file)


def data_adjust(img_file_path, label_file_path, resize=512):
    """
    调整数据尺寸 调整label标签值为0 1 2

    :param img_file_path:
    :param label_file_path:
    :param resize:
    :return:
    """
    print('[INFO]调整图片尺寸')
    img_file_list = glob(img_file_path + '*.jpg')
    label_file_list = glob(label_file_path + '*.png')
    assert len(img_file_list) == len(label_file_list)
    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img = cv2.imread(img_file)
        img = cv2.resize(img, dsize=(resize, resize))
        cv2.imwrite(img_file, img)

        label = cv2.imread(label_file, 0)
        label = 255 - label
        (rows, cols) = np.where(label == 127)
        label[rows, cols] = 1
        (rows, cols) = np.where(label == 255)
        label[rows, cols] = 2
        label = cv2.resize(label, dsize=(resize, resize), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(label_file, label)


def data_distribution(img_file_list, label_file_list, target_img_path, target_label_path):
    """
    用于拆分数据集 将文件路径列表指定文件复制到目标文件夹

    :param img_file_list:
    :param label_file_list:
    :param target_img_path:
    :param target_label_path:
    :return:
    """
    print('[INFO]分发数据')
    assert len(img_file_list) == len(label_file_list)
    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img_file_name = img_file.split('/')[-1]
        label_file_name = label_file.split('/')[-1]
        copyfile(img_file, target_img_path + img_file_name)
        copyfile(label_file, target_label_path + label_file_name)


def distribution(is_bmp_to_png, is_data_resize, config_path):
    """
    分发原始数据集
    1）获取配置文件信息 2）分发前的一些初始化工作 3）分发数据集至训练集验证集测试集 其中训练集与验证集进行了尺寸上的调整

    :param is_bmp_to_png:
    :param is_data_resize:
    :param config_path:
    :return:
    """
    config_reader = ConfigReader(config_path)
    refuge_info = config_reader.get_refuge_info()

    img_original_path = refuge_info['img_original_path']
    label_original_path = refuge_info['label_original_path']
    train_img_path = refuge_info['train_img_path']
    train_label_path = refuge_info['train_label_path']
    validation_img_path = refuge_info['validation_img_path']
    validation_label_path = refuge_info['validation_label_path']
    test_img_path = refuge_info['test_img_path']
    test_label_path = refuge_info['test_label_path']
    train_file_number_rate = min(refuge_info['train_file_number_rate'], 1)
    validation_file_number_rate = min(refuge_info['validation_file_number_rate'], 1 - train_file_number_rate)
    test_file_number_rate = min(refuge_info['test_file_number_rate'],
                                1 - train_file_number_rate - validation_file_number_rate)

    recreate_dir_list([train_img_path, train_label_path, validation_img_path, validation_label_path,
                       test_img_path, test_label_path])

    # 将label原bmp格式修改为png bmp有点大
    if is_bmp_to_png:
        label_bmp_to_png(label_original_path)

    ori_img_file_list = get_specific_type_file_list(img_original_path, 'jpg')
    ori_label_file_list = get_specific_type_file_list(label_original_path, 'png')
    assert len(ori_img_file_list) == len(ori_label_file_list)

    ori_img_file_list, ori_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)

    file_num = len(ori_img_file_list)
    train_img_file_list = ori_img_file_list[:int(file_num * train_file_number_rate)]
    train_label_file_list = ori_label_file_list[:int(file_num * train_file_number_rate)]
    validation_img_file_list = ori_img_file_list[int(file_num * train_file_number_rate):
                                                 int(file_num * (1 - test_file_number_rate))]
    validation_label_file_list = ori_label_file_list[int(file_num * train_file_number_rate):
                                                     int(file_num * (1 - test_file_number_rate))]
    test_img_file_list = ori_img_file_list[int(file_num * (1 - test_file_number_rate)):]
    test_label_file_list = ori_label_file_list[int(file_num * (1 - test_file_number_rate)):]

    data_distribution(train_img_file_list, train_label_file_list, train_img_path, train_label_path)
    data_distribution(validation_img_file_list, validation_label_file_list,
                      validation_img_path, validation_label_path)
    data_distribution(test_img_file_list, test_label_file_list, test_img_path, test_label_path)

    # 录入日志
    logger.info('train_img_file_list: ' + ','.join(train_img_file_list))
    logger.info('train_label_file_list: ' + ','.join(train_label_file_list))
    logger.info('validation_img_file_list: ' + ','.join(validation_img_file_list))
    logger.info('validation_label_file_list: ' + ','.join(validation_label_file_list))
    logger.info('test_img_file_list: ' + ','.join(test_img_file_list))
    logger.info('test_label_file_list: ' + ','.join(test_label_file_list))

    if is_data_resize:
        data_adjust(train_img_path, train_label_path, resize=512)
        data_adjust(validation_img_path, validation_label_path, resize=512)


if __name__ == '__main__':
    # is_bmp_to_png 是否将原始label由bmp格式转为png格式
    # is_data_resize 是否将训练集和验证集的图片调整为512*512的尺寸
    distribution(is_bmp_to_png=False, is_data_resize=True, config_path='../config/config.yml')
