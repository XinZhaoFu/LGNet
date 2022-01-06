import os
import argparse
import datetime
import tensorflow as tf
from tensorflow.keras import mixed_precision

from config.config_reader import ConfigReader
from model.lgnet import LGNet
from data_utils.data_loader import Data_Loader_File
import matplotlib.pyplot as plt
import pandas as pd
import setproctitle
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

setproctitle.setproctitle("xzf")

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print('[INFO] 计算类型: %s' % policy.compute_dtype)
print('[INFO] 变量类型: %s' % policy.variable_dtype)


def parseArgs():
    """
    获得参数

    :return:
    """
    parser = argparse.ArgumentParser(description='lgnet')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='learning_rate',
                        default=0,
                        type=float)
    parser.add_argument('--epochs',
                        dest='epochs',
                        help='epochs',
                        default=1,
                        type=int)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='batch_size',
                        default=4,
                        type=int)
    parser.add_argument('--load_weights',
                        dest='load_weights',
                        help='load_weights type is boolean',
                        default=False, type=bool)
    parser.add_argument('--data_augmentation',
                        dest='data_augmentation',
                        help='data_augmentation type is float, range is 0 ~ 1',
                        default=0,
                        type=float)
    args = parser.parse_args()
    return args


class Train:
    def __init__(self,
                 train_img_path,
                 train_label_path,
                 validation_img_path,
                 validation_label_path,
                 load_weights=False,
                 batch_size=128,
                 epochs=0,
                 is_data_augmentation=False,
                 augmentation_rate=1,
                 learning_rate=0,
                 ex_info='info',
                 checkpoint_save_path='./checkpoint/'):
        """

        :param train_img_path:
        :param train_label_path:
        :param validation_img_path:
        :param validation_label_path:
        :param load_weights:
        :param batch_size:
        :param epochs:
        :param is_data_augmentation:
        :param augmentation_rate:
        :param learning_rate:
        :param ex_info:
        :param checkpoint_save_path:
        """
        self.load_weights = load_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_data_augmentation = is_data_augmentation

        self.augmentation_rate = augmentation_rate
        self.learning_rate = learning_rate
        self.checkpoint_save_path = checkpoint_save_path + ex_info + '.ckpt'
        # self.checkpoint_input_path = './checkpoint/' + 'u2net_dice_02aug42000' + '.ckpt'
        self.checkpoint_input_path = self.checkpoint_save_path
        print('[INFO] checkpoint_input_path:\t' + self.checkpoint_input_path)

        self.strategy = tf.distribute.MirroredStrategy()
        print('目前使用gpu数量为: {}'.format(self.strategy.num_replicas_in_sync))
        if self.strategy.num_replicas_in_sync >= 8:
            print('[INFO]----------卡数上八!!!---------')
            sys.exit()

        data_loader = Data_Loader_File(train_img_path,
                                       train_label_path,
                                       validation_img_path,
                                       validation_label_path,
                                       batch_size,
                                       is_data_augmentation=False,
                                       data_augmentation_info=None)
        self.train_datasets = data_loader.load_train_data()
        self.val_datasets = data_loader.load_val_data()

    def model_train(self):
        """
        可多卡训练

        :return:
        """
        with self.strategy.scope():

            model = LGNet(filters_cbr=32, num_class=3, num_cbr=1, end_activation='softmax')

            if self.learning_rate > 0:
                optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
                print('[INFO] 使用sgd,其值为：\t' + str(self.learning_rate))
            else:
                optimizer = 'Adam'
                print('[INFO] 使用Adam')

            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy,
                metrics=[tf.keras.metrics.MeanIoU(num_classes=2), 'categorical_accuracy']
            )

            if os.path.exists(self.checkpoint_input_path + '.index') and self.load_weights:
                print("[INFO] -------------------------------------------------")
                print("[INFO] -----------------loading weights-----------------")
                print("[INFO] -------------------------------------------------")
                model.load_weights(self.checkpoint_input_path)

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_save_path,
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                save_freq='epoch')

            history = model.fit(
                self.train_datasets,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.val_datasets,
                validation_freq=1,
                callbacks=[checkpoint_callback]
            )

            if self.epochs == 1:
                # 一般都是训练前专门看一下信息 所以正常训练时就不显示了 主要还是tmux不能上翻 有的时候会遮挡想看的信息
                model.summary()

        return history


def plot_learning_curves(history, plt_name):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig('./log/' + plt_name + '.jpg')


def train_init(config_path='../config/config.yml'):
    """
    初始化参数

    :return:
    """
    config_reader = ConfigReader(config_path)

    refuge_info = config_reader.get_refuge_info()
    train_img_path = refuge_info['train_img_path']
    train_label_path = refuge_info['train_label_path']
    validation_img_path = refuge_info['validation_img_path']
    validation_label_path = refuge_info['validation_label_path']

    ex_info = 'u2net_dice_02aug42000'

    print('[INFO] 实验名称：' + ex_info)

    start_time = datetime.datetime.now()

    tran_tab = str.maketrans('- :.', '____')
    plt_name = ex_info + str(start_time).translate(tran_tab)

    args = parseArgs()
    seg = Train(load_weights=args.load_weights,
                batch_size=args.batch_size,
                epochs=args.epochs,
                data_augmentation=args.data_augmentation,
                learning_rate=args.learning_rate,
                ex_info=ex_info)

    history = seg.model_train()
    plot_learning_curves(history, plt_name)

    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])


if __name__ == '__main__':
    train_init()
