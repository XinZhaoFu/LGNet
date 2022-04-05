import os
import shutil
from os import remove
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob

from model.utils import LGNet_Utils


def create_dir(folder_name):
    """
    创建文件夹

    :param folder_name:
    :return: None
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('[INFO] 新建文件夹：' + folder_name)
    else:
        print('[INFO] 已存在文件夹：' + folder_name)


def recreate_dir(folder_name):
    """
    重建文件夹

    :param folder_name:
    :return: None
    """
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    print('[INFO] 重建文件夹：' + folder_name)


def recreate_dir_list(dir_list):
    """
    对列表内所有文件夹重建

    :param dir_list:
    :return: None
    """
    for folder_name in dir_list:
        recreate_dir(folder_name)


def distribution_file(file_path_list, target_file_path, is_recreate_dir=False):
    """
    将文件路径列表中的文件复制到目标文件夹

    :param is_recreate_dir:
    :param file_path_list:
    :param target_file_path:
    :return:
    """
    if is_recreate_dir:
        recreate_dir(target_file_path)

    for file_path in tqdm(file_path_list):
        file_name = file_path.split('/')[-1]
        shutil.copyfile(file_path, target_file_path + file_name)
    print('[INFO] 目标文件夹：' + target_file_path + ' 文件传输已完成')


def shuffle_file(img_file_list, label_file_list):
    """
    打乱img和label的文件列表顺序 但对位文件顺序不变 并返回两列表 seed已固定以此保证实验可重复性

    :param img_file_list:
    :param label_file_list:
    :return:
    """
    assert len(img_file_list) == len(label_file_list)
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    return img_file_list.tolist(), label_file_list.tolist()


def get_specific_type_file_list(file_path, file_type):
    """
    对glob进行了一次封装 旨在避免windows下偶尔会出现反斜杠与斜杠混用的情况

    :param file_path:
    :param file_type:
    :return:
    """
    file_list = glob(file_path + '*.' + file_type)
    if len(file_list) > 0 and '\\' in file_list[0]:
        file_list = [file.replace('\\', '/') for file in file_list]

    return file_list


def file_consistency_check(img_file_list, label_file_list):
    """
    校验文件是否对应

    :param img_file_list:
    :param label_file_list:
    :return flag:
    """
    flag = True
    assert len(img_file_list) == len(label_file_list)
    for img_path, label_path in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img_name = (img_path.split('/')[-1]).split('.')[0]
        label_name = (label_path.split('/')[-1]).split('.')[0]
        if img_name != label_name:
            # print(img_name, label_name)
            flag = False
            break
    if flag:
        print('[INFO] 文件一致性校验通过')
    else:
        print('[INFO] 文件一致性校验未通过')

    return flag


def datasets_check(train_img_file_list,
                   train_label_file_list,
                   validation_img_file_list,
                   validation_label_file_list,
                   test_img_file_list,
                   test_label_file_list, is_reload=True):
    """

    :param train_img_file_list:
    :param train_label_file_list:
    :param validation_img_file_list:
    :param validation_label_file_list:
    :param test_img_file_list:
    :param test_label_file_list:
    :param is_reload:
    :return:
    """
    if not file_consistency_check(train_img_file_list, train_label_file_list):
        train_img_file_list.sort()
        train_label_file_list.sort()
    if not file_consistency_check(validation_img_file_list, validation_label_file_list):
        validation_img_file_list.sort()
        validation_label_file_list.sort()
    if not file_consistency_check(test_img_file_list, test_label_file_list):
        test_img_file_list.sort()
        test_label_file_list.sort()

    if is_reload:
        train_img_file_list.extend(test_img_file_list)
        train_label_file_list.extend(test_label_file_list)

    return train_img_file_list, train_label_file_list, validation_img_file_list, validation_label_file_list, test_img_file_list, test_label_file_list



def data_adjust(img_file_path, label_file_path, is_resize=True, is_center=False):
    """
    调整label灰度值为类别码

    :param img_file_path:
    :param label_file_path:
    :param is_resize:
    :param is_center:
    :return:
    """
    img_file_list = glob(img_file_path + '*.jpg')
    label_file_list = glob(label_file_path + '*.png')
    if len(label_file_list) == 0:
        label_file_list = glob(label_file_path + '*.bmp')
    assert len(img_file_list) == len(label_file_list)

    for img_file, label_file in tqdm(zip(img_file_list, label_file_list), total=len(img_file_list)):
        img = cv2.imread(img_file)
        ori_label = cv2.imread(label_file, 0)

        if is_center:
            img, ori_label = center_crop(img, ori_label)

        label = np.zeros(shape=ori_label.shape, dtype=np.uint8)

        (rows, cols) = np.where(ori_label == 127)
        label[rows, cols] = 1
        (rows, cols) = np.where(ori_label == 0)
        label[rows, cols] = 2

        if is_resize:
            img = cv2.resize(img, (512, 512))
            label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(img_file, img)
        cv2.imwrite(label_file, label)


def one_data_adjust(ori_img, ori_label):
    """
    调整label灰度值为类别码
    调整图像尺寸为512

    :param ori_img:
    :param ori_label:
    :return:
    """
    label = np.zeros(shape=ori_label.shape, dtype=np.uint8)

    (rows, cols) = np.where(ori_label == 127)
    label[rows, cols] = 1
    (rows, cols) = np.where(ori_label == 0)
    label[rows, cols] = 2

    img = cv2.resize(ori_img, (512, 512))
    label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

    return img, label


def reverse_one_data_adjust(ori_label):
    """
    one_data_adjust反置 但未设置尺寸调节
    :param ori_label:
    :return:
    """
    label = np.zeros(shape=ori_label.shape, dtype=np.uint8)

    (rows, cols) = np.where(ori_label == 1)
    label[rows, cols] = 128
    (rows, cols) = np.where(ori_label == 2)
    label[rows, cols] = 255

    return label


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


def get_temp_info(img, temp_path):
    """

    :param img:
    :param temp_path:
    :return:
    """
    temp = cv2.imread('./datasets/refuge_datasets/test/label/' + temp_path + '.png', 0)
    # print('./datasets/refuge_datasets/test/label/' + temp_path + '.png')
    rows, cols = np.where(temp == 255)
    center_row = int(sum(rows) / len(rows))
    center_col = int(sum(cols) / len(cols))
    crop_row_top, crop_row_down, crop_col_right, crop_col_left = max(center_row - 256, 0), center_row + 256, max(center_col - 256, 0), center_col + 256
    temp_img = np.zeros(shape=(crop_row_down - crop_row_top, crop_col_left - crop_col_right, 3), dtype=np.uint8)
    temp_img[:, :, :] = img[crop_row_top:crop_row_down, crop_col_right:crop_col_left, :]
    cv2.imwrite('./datasets/refuge_datasets/temp/' + temp_path + '.jpg', temp_img)

    return temp_img, [crop_row_top, crop_row_down, crop_col_right, crop_col_left]


def img_init_utils(test_file):
    """

    :param test_file:
    :param is_zero:
    :return:
    """
    img = cv2.imread(test_file)
    temp_img, temp_info = get_temp_info(img, (test_file.split('/')[-1]).split('.')[0])

    return temp_img, temp_info


def test_img_init(test_img, is_zero=False):
    """

    :param test_img:
    :param is_zero:
    :return:
    """
    test_img, init_info = img_init_utils(test_img)
    test_img = cv2.resize(test_img, dsize=(512, 512))

    if is_zero:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img = np.array(test_img, dtype=np.float) / 255.
        test_img_np = np.empty((1, 512, 512, 1))
        test_img_np[0, :, :, 0] = test_img
        test_img = test_img_np
    else:
        test_img = np.array(test_img, dtype=np.float) / 255.
        test_img = np.reshape(test_img, newshape=(1, 512, 512, 3))

    return test_img, init_info


def center_crop(img, label):
    """

    :param img:
    :param label:
    :return:
    """
    rows, cols = np.where(label == 255)
    center_row = int(sum(rows) / len(rows))
    center_col = int(sum(cols) / len(cols))
    crop_row_top, crop_row_down, crop_col_right, crop_col_left = max(center_row-256, 0), center_row+256, max(center_col-256, 0), center_col+256

    crop_img = np.zeros(shape=(crop_row_down - crop_row_top, crop_col_left - crop_col_right, 3), dtype=np.uint8)
    crop_img[:, :, :] = img[crop_row_top:crop_row_down, crop_col_right:crop_col_left, :]
    crop_label = np.zeros(shape=(crop_row_down - crop_row_top, crop_col_left - crop_col_right), dtype=np.uint8)
    crop_label[:, :] = label[crop_row_top:crop_row_down, crop_col_right:crop_col_left]

    return crop_img, crop_label


def model_check(model):
    """"""

    model1 = LGNet_Utils()
    model1.load_weights('./checkpoint/lgnet_refuge_16490389479179022/lgnet_refuge_16490389479179022.ckpt')
    model2 = LGNet_Utils()
    model2.load_weights('./checkpoint/lgnet_refuge_1649038847275365/lgnet_refuge_1649038847275365.ckpt')

    return model, model1, model2


def inference_check(model, checkpoint_save_path):
    """

    :param model:
    :param checkpoint_save_path:
    :return:
    """
    _, model1, model2 = model_check(model)
    if os.access(checkpoint_save_path, os.F_OK) and os.access(checkpoint_save_path, os.R_OK):
        print('[INFO] 检测通过')

    return [model, model1, model2]


def get_inference_img(models, test_file):
    """

    :param models:
    :param test_file:
    :return:
    """
    test_img, init_info = test_img_init(test_file, is_zero=True)

    predictd = models[1].predict(test_img)
    predictd = tf.math.argmax(predictd, 3)
    predictd = np.array(predictd)
    predictd = np.reshape(predictd, newshape=(512, 512))

    predict_img = np.zeros(shape=(512, 512), dtype=np.uint8)
    predict_img[:, :] = predictd[:, :] * 127

    predictc = models[2].predict(test_img)
    predictc = tf.math.argmax(predictc, 3)
    predictc = np.array(predictc)
    predictc = np.reshape(predictc, newshape=(512, 512))

    rows, cols = np.where(predictc == 1)
    predict_img[rows, cols] = 255

    out = np.zeros(shape=(1634, 1634), dtype=np.uint8)
    predict_img = cv2.resize(predict_img, dsize=(init_info[3] - init_info[2], init_info[1] - init_info[0]))
    out[init_info[0]:init_info[1], init_info[2]:init_info[3]] = predict_img

    return out


def binary_predict_value_to_img(predict_value, img_width=512):
    """
    将独热码存储的图像数据转为常规图像值 默认该图为全0 独热码的第二个通道是该像素为1的概率 大于0.5则设定为1
    :param predict_value:
    :param img_width:
    :return:
    """
    predict_img_temp = np.zeros((512, 512, 1))
    for i in range(img_width):
        for j in range(img_width):
            if predict_value[i, j, 1] > 0.5:
                predict_img_temp[i, j, 0] = 1
    binary_predict_img = predict_img_temp * 255.

    return binary_predict_img


def get_oc_od(predict):
    """

    :param predict:
    :return:
    """

    od_predict = np.zeros(shape=(1634, 1634), dtype=np.uint8)
    rows, cols = np.where(predict == 127)
    od_predict[rows, cols] = 1

    oc_predict = np.zeros(shape=(1634, 1634), dtype=np.uint8)
    rows, cols = np.where(predict == 255)
    oc_predict[rows, cols] = 1

    return od_predict, oc_predict