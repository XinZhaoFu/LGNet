import os
import shutil
from os import remove
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


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
    return img_file_list, label_file_list


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
            print(img_name, label_name)
            flag = False
            break
    if flag:
        print('[INFO] 文件一致性校验通过')
    else:
        print('[INFO] 文件一致性校验未通过')

    return flag


def data_adjust(img_file_path, label_file_path, is_resize=True):
    """
    调整label灰度值为类别码
    调整图像尺寸为512

    :param img_file_path:
    :param label_file_path:
    :param is_resize: 是否调整图片尺寸  测试集不需要
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
        label = np.zeros(shape=ori_label.shape, dtype=np.uint8)

        (rows, cols) = np.where(ori_label == 128)
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

    (rows, cols) = np.where(ori_label == 128)
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


def img_init_utils(img):
    """

    :param img:
    :return:
    """
    img_size, _, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, max_val, _, max_loc = cv2.minMaxLoc(img_gray)

    crop_row_top, crop_row_down = max(max_loc[1] - 256, 0), min(max_loc[1] + 256, img_size)
    crop_col_right, crop_col_left = max(max_loc[0] - 256, 0), min(max_loc[0] + 256, img_size)
    # print(crop_row_top, crop_row_down, crop_col_right, crop_col_left)
    position_img = np.zeros(shape=(crop_row_down - crop_row_top, crop_col_left - crop_col_right, 3), dtype=np.uint8)
    position_img[:, :, :] = img[crop_row_top:crop_row_down, crop_col_right:crop_col_left, :]
    return position_img, [crop_row_top, crop_row_down, crop_col_right, crop_col_left]


def test_img_init(test_img):
    """

    :param test_img:
    :return:
    """
    test_img, init_info = img_init_utils(test_img)
    test_img = cv2.resize(test_img, dsize=(512, 512))
    test_img = np.array(test_img / 255.)
    test_img = np.reshape(test_img, newshape=(1, 512, 512, 3))

    return test_img, init_info
