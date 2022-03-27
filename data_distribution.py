import datetime
from shutil import copyfile
from config.config_reader import ConfigReader
from tqdm import tqdm
from data_utils.utils import shuffle_file, recreate_dir_list, data_adjust, label_bmp_to_png, \
    get_specific_type_file_list, file_consistency_check
from loguru import logger
from data_utils.augmentation_utils import augmentation_distribution


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


def distribution(is_bmp_to_png=False,
                 config_path='./config/config.yml',
                 train_file_rate=.6,
                 test_file_rate=.2):
    """

    :param is_bmp_to_png:是否将原始label由bmp格式转为png格式
    :param config_path:
    :param train_file_rate:
    :param test_file_rate:
    :return:
    """
    config_reader = ConfigReader(config_path)
    refuge_info = config_reader.get_refuge_info()

    img_original_path = refuge_info['img_original_path']
    label_original_path = refuge_info['label_original_path']
    train_img_path = refuge_info['train_img_path']
    train_label_path = refuge_info['train_label_path']
    train_aug_img_path = refuge_info['train_aug_img_path']
    train_aug_label_path = refuge_info['train_aug_label_path']
    validation_img_path = refuge_info['validation_img_path']
    validation_label_path = refuge_info['validation_label_path']
    test_img_path = refuge_info['test_img_path']
    test_label_path = refuge_info['test_label_path']
    augmentation_rate = refuge_info['augmentation_rate']

    recreate_dir_list([train_img_path, train_label_path,
                       train_aug_img_path, train_aug_label_path,
                       validation_img_path, validation_label_path,
                       test_img_path, test_label_path])

    # 将label原bmp格式修改为png bmp有点大
    if is_bmp_to_png:
        label_bmp_to_png(label_original_path)

    ori_img_file_list = get_specific_type_file_list(img_original_path, 'jpg')
    ori_label_file_list = get_specific_type_file_list(label_original_path, 'png')
    print(len(ori_img_file_list), len(ori_label_file_list))
    assert len(ori_img_file_list) == len(ori_label_file_list)

    if not file_consistency_check(ori_img_file_list, ori_label_file_list):
        ori_img_file_list.sort()
        ori_label_file_list.sort()
    ori_img_file_list, ori_label_file_list = shuffle_file(ori_img_file_list, ori_label_file_list)
    if not file_consistency_check(ori_img_file_list, ori_label_file_list):
        exit(0)

    file_num = len(ori_img_file_list)
    train_img_file_list = ori_img_file_list[:int(file_num * train_file_rate)]
    train_label_file_list = ori_label_file_list[:int(file_num * train_file_rate)]

    validation_img_file_list = ori_img_file_list[int(file_num * train_file_rate):
                                                 int(file_num * (1 - test_file_rate))]
    validation_label_file_list = ori_label_file_list[int(file_num * train_file_rate):
                                                     int(file_num * (1 - test_file_rate))]
    test_img_file_list = ori_img_file_list[int(file_num * (1 - test_file_rate)):]
    test_label_file_list = ori_label_file_list[int(file_num * (1 - test_file_rate)):]

    data_distribution(train_img_file_list, train_label_file_list, train_img_path, train_label_path)
    data_distribution(validation_img_file_list, validation_label_file_list,
                      validation_img_path, validation_label_path)
    data_distribution(test_img_file_list, test_label_file_list, test_img_path, test_label_path)
    augmentation_distribution(train_img_file_list, train_label_file_list, train_aug_img_path, train_aug_label_path, augmentation_rate)

    # 录入日志
    logger.info('train_img_file_list: ' + ','.join(train_img_file_list))
    logger.info('train_label_file_list: ' + ','.join(train_label_file_list))
    logger.info('validation_img_file_list: ' + ','.join(validation_img_file_list))
    logger.info('validation_label_file_list: ' + ','.join(validation_label_file_list))
    logger.info('test_img_file_list: ' + ','.join(test_img_file_list))
    logger.info('test_label_file_list: ' + ','.join(test_label_file_list))

    data_adjust(train_img_path, train_label_path)
    data_adjust(validation_img_path, validation_label_path)
    data_adjust(test_img_path, test_label_path, is_resize=False)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    distribution()

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])
