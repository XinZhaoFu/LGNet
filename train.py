import os
import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision
from config.config_reader import ConfigReader
from model.lgnet import LGNet
from model.unetm import UNetM
from data_utils.data_loader import Data_Loader_File
import setproctitle
import sys
import numpy as np
from time import time
from model.loss import categorical_crossentropy_weight, dice_loss, mix_dice_focal_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

setproctitle.setproctitle('阿福给您拜个早年')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# gpus = tf.config.list_physical_devices('GPU')
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('[INFO] 计算类型: %s' % policy.compute_dtype)
print('[INFO] 变量类型: %s' % policy.variable_dtype)


class Train:
    def __init__(self,
                 train_img_path,
                 train_label_path,
                 validation_img_path,
                 validation_label_path,
                 is_load_weight,
                 batch_size,
                 epochs,
                 learning_rate,
                 experiment_name,
                 checkpoint_save_path,
                 checkpoint_input_path,
                 model_name,
                 optimizers,
                 num_class
                 ):
        """

        :param train_img_path:
        :param train_label_path:
        :param validation_img_path:
        :param validation_label_path:
        :param is_load_weight:
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :param experiment_name:
        :param checkpoint_save_path:
        :param checkpoint_input_path:
        :param optimizers:
        :param num_class:
        """
        self.load_weights = is_load_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizers = optimizers
        self.learning_rate = learning_rate
        self.checkpoint_save_path = checkpoint_save_path + experiment_name + '.ckpt'
        self.checkpoint_input_path = checkpoint_input_path
        self.model_name = model_name
        self.num_class = num_class

        self.strategy = tf.distribute.MirroredStrategy()
        print('目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))
        if self.strategy.num_replicas_in_sync >= 8:
            print('[INFO]----------卡数上八!!!---------')
            sys.exit()

        data_loader = Data_Loader_File(train_img_path=train_img_path,
                                       train_label_path=train_label_path,
                                       validation_img_path=validation_img_path,
                                       validation_label_path=validation_label_path,
                                       batch_size=batch_size)
        self.train_datasets = data_loader.load_train_data()
        self.val_datasets = data_loader.load_val_data()

    def model_train(self):
        """
        可多卡训练

        :return:
        """
        with self.strategy.scope():

            # 这里最好用match case 看着会更舒服一些  但是升3.10包冲突太多就算了
            # if self.model_name == 'lgnet':
            #     model = LGNet(filters=32, num_class=self.num_class)
            # else:
            #     print('[INFO]模型数据有误')
            #     sys.exit()

            model = LGNet(filters=32, num_class=self.num_class)
            # model = UNetM(filters=32, num_class=self.num_class)

            optimizer = 'Adam'
            if self.optimizers == 'Adam':
                print('[INFO] 使用Adam')
            elif self.optimizers == 'SGD':
                optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
                print('[INFO] 使用SGD,其值为：\t' + str(self.learning_rate))
            else:
                print('[INFO] 未设置优化器 默认使用Adam')

            model.compile(
                optimizer=optimizer,
                # loss=categorical_crossentropy_weight(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                # loss_weights=[1.0, 10.0, 10.0],
                # metrics=[tf.keras.metrics.MeanIoU(num_classes=self.num_class)]
                metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.num_class)])

            if os.path.exists(self.checkpoint_input_path + '.index') and self.load_weights:
                print("[INFO] -------------------------------------------------")
                print("[INFO] -----------------loading weights-----------------")
                print("[INFO] -------------------------------------------------")
                print('[INFO] checkpoint_input_path:\t' + self.checkpoint_input_path)
                model.load_weights(filepath=self.checkpoint_input_path)

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_save_path,
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                save_freq='epoch'
            )

            history = model.fit(
                self.train_datasets,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.val_datasets,
                validation_freq=1,
                callbacks=[checkpoint_callback],
            )

            if self.epochs == 1:
                # 一般都是训练前专门看一下信息 所以正常训练时就不显示了
                model.summary()

        return history


def train_init(config_path='./config/config.yml'):
    """
    初始化参数

    :return:
    """
    start_time = datetime.datetime.now()

    config_reader = ConfigReader(config_path)

    refuge_info = config_reader.get_refuge_info()
    train_img_path = refuge_info['train_img_path']
    train_label_path = refuge_info['train_label_path']
    validation_img_path = refuge_info['validation_img_path']
    validation_label_path = refuge_info['validation_label_path']
    train_aug_img_path = refuge_info['train_aug_img_path']
    train_aug_label_path = refuge_info['train_aug_label_path']
    train_info = config_reader.get_train_info()
    experiment_name = train_info['experiment_name']
    is_load_weight = train_info['is_load_weight']
    batch_size = train_info['batch_size']
    epochs = train_info['epochs']
    is_data_augmentation = train_info['is_data_augmentation']
    learning_rate = train_info['learning_rate']
    checkpoint_save_path = train_info['checkpoint_save_path']
    checkpoint_input_path = train_info['checkpoint_input_path']
    optimizers = train_info['optimizers']
    num_class = train_info['num_class']
    model_name = train_info['model_name']

    if is_data_augmentation:
        train_img_path, train_label_path = train_aug_img_path, train_aug_label_path

    time_str = str(time()).replace('.', '')
    experiment_name = experiment_name + '_' + time_str
    print('[INFO] 实验名称：' + experiment_name)

    seg = Train(train_img_path,
                train_label_path,
                validation_img_path,
                validation_label_path,
                is_load_weight,
                batch_size,
                epochs,
                learning_rate,
                experiment_name,
                checkpoint_save_path,
                checkpoint_input_path,
                model_name,
                optimizers,
                num_class)

    history = seg.model_train()

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    train_init()
