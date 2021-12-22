from glob import glob
import cv2
from config.config_reader import ConfigReader
from tqdm import tqdm
from data_utils.utils import shuffle_file, recreate_dir_list


def label_bmp_to_png(label_path):
    label_file_list = glob(label_path + '*.bmp')
    print('[INFO]载入bmp格式label 共计：' + str(len(label_file_list)))
    for label_file in tqdm(label_file_list):
        label = cv2.imread(label_file)
        cv2.imwrite(label_file[:-3] + 'png', label)

def data_distribution():



if __name__ == '__main__':
    config_reader = ConfigReader(config_path='../config/config.yml')
    refuge_data = config_reader.get_refuge_data()

    img_original_path = refuge_data['img_original_path']
    label_original_path = refuge_data['label_original_path']
    train_img_path = refuge_data['train_img_path']
    train_label_path = refuge_data['train_label_path']
    validation_img_path = refuge_data['validation_img_path']
    validation_label_path = refuge_data['validation_label_path']
    test_img_path = refuge_data['test_img_path']
    test_label_path = refuge_data['test_label_path']
    train_file_number_rate = min(refuge_data['train_file_number_rate'], 1)
    validation_file_number_rate = min(refuge_data['validation_file_number_rate'], 1 - train_file_number_rate)
    test_file_number_rate = min(refuge_data['test_file_number_rate'],
                                1 - train_file_number_rate - validation_file_number_rate)

    recreate_dir_list([train_img_path, train_label_path, validation_img_path, validation_label_path,
                       test_img_path, test_label_path])

    is_bmp_to_png = False
    # 将label原bmp格式修改为png bmp有点大
    if is_bmp_to_png:
        label_bmp_to_png(label_original_path)


