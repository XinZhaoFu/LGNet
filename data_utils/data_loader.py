from data_utils.utils import get_specific_type_file_list, shuffle_file, file_consistency_check


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


    return 0
