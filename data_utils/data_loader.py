from data_utils.utils import shuffle_file, file_consistency_check, get_specific_type_file_list
import tensorflow as tf


class Data_Loader_File:
    def __init__(self,
                 train_img_path,
                 train_label_path,
                 validation_img_path,
                 validation_label_path,
                 batch_size):
        """
        构建数据管道

        :param train_img_path:
        :param train_label_path:
        :param validation_img_path:
        :param validation_label_path:
        :param batch_size:
        """
        self.train_img_path = train_img_path
        self.train_label_path = train_label_path
        self.validation_img_path = validation_img_path
        self.validation_label_path = validation_label_path
        self.batch_size = batch_size

    def load_train_data(self):
        print('[INFO] 正在载入训练集')
        train_datasets = _data_preprocess(self.train_img_path,
                                          self.train_label_path,
                                          self.batch_size)
        return train_datasets

    def load_val_data(self):
        print('[INFO] 正在载入验证集')
        validation_datasets = _data_preprocess(self.validation_img_path,
                                               self.validation_label_path,
                                               self.batch_size)
        return validation_datasets


def _data_preprocess(img_file_path,
                     label_file_path,
                     batch_size):
    """

    :param img_file_path:
    :param label_file_path:
    :param batch_size:
    :return:
    """
    img_file_list = get_specific_type_file_list(img_file_path, 'jpg')
    label_file_list = get_specific_type_file_list(label_file_path, 'png')

    assert len(img_file_list) > 0
    assert len(img_file_list) == len(label_file_list)

    img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)

    if not file_consistency_check(img_file_list, label_file_list):
        print('[info]重新整理文件')
        img_file_list.sort()
        label_file_list.sort()
        img_file_list, label_file_list = shuffle_file(img_file_list, label_file_list)
    if not file_consistency_check(img_file_list, label_file_list):
        exit(0)

    print('[INFO]载入数据量：' + str(len(img_file_list)))

    datasets = tf.data.Dataset.from_tensor_slices((img_file_list, label_file_list))
    datasets = datasets.map(_load_and_preprocess_datasets, num_parallel_calls=tf.data.AUTOTUNE)
    datasets = datasets.shuffle(buffer_size=batch_size * 8)
    datasets = datasets.batch(batch_size=batch_size)
    datasets = datasets.prefetch(buffer_size=tf.data.AUTOTUNE)

    return datasets


def _load_and_preprocess_datasets(img_path, label_path):
    """
    对img和label进行读取预处理 其中label以onehot的形式

    :param img_path:
    :param label_path:
    :return:
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.reshape(tensor=image, shape=(512, 512))
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.reshape(label, [512, 512])
    # label = tf.reshape(tensor=label, shape=(512, 512))

    label = tf.cast(label, dtype=tf.uint8)
    label = tf.one_hot(indices=label, depth=3)
    print(image.shape, label.shape)

    return image, label
