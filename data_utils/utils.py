import os
import shutil
import numpy as np
from tqdm import tqdm


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
    for folder_name in tqdm(dir_list):
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
    打乱img和label的文件列表顺序 并返回两列表 seed已固定

    :param img_file_list:
    :param label_file_list:
    :return:
    """
    np.random.seed(10)
    index = [i for i in range(len(img_file_list))]
    np.random.shuffle(index)
    img_file_list = np.array(img_file_list)[index]
    label_file_list = np.array(label_file_list)[index]
    return img_file_list, label_file_list
