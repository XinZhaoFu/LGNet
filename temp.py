# from datetime import datetime
# import tensorflow as tf
# from data_utils.augmentation_utils import augmentation_process
# import cv2
# import time
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from data_utils.metrics_utils import get_vcdr
from data_utils.utils import get_specific_type_file_list, img_init_utils, reverse_one_data_adjust

label = cv2.imread('./datasets/refuge_datasets/test/label/V0005.png', 0)
vcdr = get_vcdr(label)
print(vcdr)

# # label_file_list = get_specific_type_file_list('./datasets/refuge_datasets/validation/label/', 'png')
# label_file_list = get_specific_type_file_list('./datasets/refuge_datasets/test/label/', 'png')
# # label_file_list = label_file_list[:40]
#
# for label_file in label_file_list:
#     label = cv2.imread(label_file, 0)
#     label_name = label_file.split('/')[-1]
#     re_label = reverse_one_data_adjust(label)
#     cv2.imwrite('./datasets/refuge_datasets/temp/' + label_name, re_label)

# label = cv2.imread('./datasets/refuge_datasets/temp/V0001_00.png', 0)
# print(label)


# img_file_list = get_specific_type_file_list('./datasets/refuge_datasets/test/img/', 'jpg')
# for img_file in img_file_list:
#     img = cv2.imread(img_file)
#     img_name = img_file.split('/')[-1]
#     crop_img, _ = img_init_utils(img)
#     cv2.imwrite('./datasets/refuge_datasets/temp/' + img_name, crop_img)

# for _ in range(100):
#     choice_list = [np.random.randint(0, 2) for _ in range(4)]
#     print(choice_list)

# label_list = get_specific_type_file_list('./datasets/refuge_datasets/train/label/', 'png')

# img_list.sort()
# label_list.sort()
# print(img_list)
# print(label_list)

# img = cv2.imread('./datasets/refuge_datasets/train/img/V0001.jpg')
# label = cv2.imread('./datasets/refuge_datasets/train/label/V0010.png', 0)
# img_list, label_list = augmentation_process(img, label, 9)


# start_time = datetime.now()
#
# start_time = str(start_time)[:19]
# tran_tab = str.maketrans('- :', '___')
# plt_name = str(start_time).translate(tran_tab)
#
# print(plt_name, start_time)

# x = tf.constant([[[1, 2, 3, 4, 5, 6], [2, 2, 3, 4, 5, 6]], [[3, 2, 3, 4, 5, 6], [4, 2, 3, 4, 5, 6]]])
# print(x)
# temp = tf.reshape(x, [2, 2, 2, 3])
# print(temp)
# temp = tf.transpose(temp, perm=[0, 1, 3, 2])
# print(temp)
# temp = tf.reshape(temp, [2, 2, 6])
# print(temp)

# temp = tf.constant([[1, 2, 3, 4, 5, 6]])
# print(temp.shape.as_list())
# for _ in range(3):
#     temp = tf.reshape(temp, [-1, 3])
#     print(temp)
#     temp = tf.transpose(temp, perm=[1, 0])
#     print(temp)
#     temp = tf.reshape(temp, [6])
#     print(temp)


