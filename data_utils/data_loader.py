from data_utils.utils import get_specific_type_file_list, shuffle_file, file_consistency_check
import tensorflow as tf


class Data_Loader_File:
    def __init__(self,
                 train_img_file_path,
                 train_label_file_path,
                 validation_img_file_path,
                 validation_label_file_path,
                 batch_size,
                 is_data_augmentation=False,
                 data_augmentation_info=None):
        """
        构建数据管道

        :param train_img_file_path:
        :param train_label_file_path:
        :param validation_img_file_path:
        :param validation_label_file_path:
        :param batch_size:
        :param is_data_augmentation:
        :param data_augmentation_info:
        """
        self.train_img_file_path = train_img_file_path
        self.train_label_file_path = train_label_file_path
        self.validation_img_file_path = validation_img_file_path
        self.validation_label_file_path = validation_label_file_path
        self.batch_size = batch_size
        self.is_data_augmentation = is_data_augmentation
        self.data_augmentation_info = data_augmentation_info

    def load_train_data(self):
        print('[INFO] 正在载入训练集')
        train_datasets = _data_preprocess(self.train_img_file_path,
                                          self.train_label_file_path,
                                          self.batch_size,
                                          self.is_data_augmentation,
                                          self.data_augmentation_info)
        return train_datasets

    def load_val_data(self):
        print('[INFO] 正在载入验证集')
        validation_datasets = _data_preprocess(self.validation_img_file_path,
                                               self.validation_label_file_path,
                                               self.batch_size)
        return validation_datasets


def _data_preprocess(img_file_path,
                     label_file_path,
                     batch_size,
                     is_data_augmentation=False,
                     data_augmentation_info=None):
    """

    :param img_file_path:
    :param label_file_path:
    :param batch_size:
    :param is_data_augmentation:
    :param data_augmentation_info:
    :return:
    """
    img_file_list = get_specific_type_file_list(img_file_path, 'jpg')
    label_file_list = get_specific_type_file_list(label_file_path, 'png')

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)
    if not file_consistency_check(img_file_list, label_file_list):
        img_file_list.sort()
        label_file_list.sort()
        img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    datasets = tf.data.Dataset.from_tensor_slices((img_file_list, label_file_list))
    datasets = datasets.map(_load_and_preprocess_onehot_datasets, num_parallel_calls=tf.data.AUTOTUNE)
    datasets = datasets.shuffle(buffer_size=batch_size * 8)
    datasets = datasets.batch(batch_size=batch_size)
    datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)

    return datasets


def _load_and_preprocess_onehot_datasets(img_path, label_path):
    """
    对img和label进行读取预处理 其中label以onehot的形式

    :param img_path:
    :param label_path:
    :return:
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.reshape(tensor=image, shape=(512, 512))
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.reshape(tensor=label, shape=(512, 512))

    label = tf.cast(label, dtype=tf.uint8)
    label = tf.one_hot(indices=label, depth=3)

    return image, label
