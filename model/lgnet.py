from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, GlobalAveragePooling2D, multiply
from model.utils import Con_Bn_Act, DW_Con_Bn_Act, channel_shuffle, Aspp, Up_CBR_Block
import tensorflow as tf


class LGNet(Model):
    def __init__(self, filters=32, num_class=3):
        super(LGNet, self).__init__()

        self.filters = filters
        self.num_class = num_class

        self.en_cbr1 = Con_Bn_Act(filters=self.filters, name='en_cbr1')
        self.asb1 = ASB(filters=self.filters, block_name='asb1', inputs_size=512)
        self.en_cbr2 = Con_Bn_Act(filters=self.filters, strides=2, name='en_cbr2')
        self.asb2 = ASB(filters=self.filters, block_name='asb2', inputs_size=256)
        self.en_cbr3 = Con_Bn_Act(filters=self.filters, strides=2, name='en_cbr3')
        self.asb3 = ASB(filters=self.filters, block_name='asb3', inputs_size=128)
        self.en_cbr4 = Con_Bn_Act(filters=self.filters, strides=2, name='en_cbr4')
        self.asb4 = ASB(filters=self.filters, block_name='asb4', inputs_size=64)
        self.en_cbr5 = Con_Bn_Act(filters=self.filters, strides=2, name='en_cbr5')

        self.aspp = Aspp(filters=self.filters, dila_rate1=6, dila_rate2=12, dila_rate3=18, dila_rate4=24)

        self.de_cbr_up5 = Up_CBR_Block(filters=self.filters, block_name='de_cbr_up5')
        self.de_cbr_up4 = Up_CBR_Block(filters=self.filters, block_name='de_cbr_up4')
        self.de_cbr_up3 = Up_CBR_Block(filters=self.filters, block_name='de_cbr_up3')
        self.de_cbr_up2 = Up_CBR_Block(filters=self.filters, block_name='de_cbr_up2')
        self.de_cbr1 = Con_Bn_Act(filters=self.num_class, activation='softmax', name='de_cbr1')

    def call(self, inputs, training=None, mask=None):
        en_cbr1 = self.en_cbr1(inputs)
        asb1 = self.asb1(en_cbr1)
        en_cbr2 = self.en_cbr2(asb1)
        asb2 = self.asb2(en_cbr2)
        en_cbr3 = self.en_cbr3(asb2)
        asb3 = self.asb3(en_cbr3)
        en_cbr4 = self.en_cbr4(asb3)
        asb4 = self.asb4(en_cbr4)
        en_cbr5 = self.en_cbr5(asb4)

        aspp = self.aspp(en_cbr5)

        de_cbr_up5 = self.de_cbr_up5(aspp)
        de_cbr_up4 = self.de_cbr_up4(concatenate([asb4, de_cbr_up5], axis=3))
        de_cbr_up3 = self.de_cbr_up3(concatenate([asb3, de_cbr_up4], axis=3))
        de_cbr_up2 = self.de_cbr_up2(concatenate([asb2, de_cbr_up3], axis=3))
        out = self.de_cbr1(concatenate([asb1, de_cbr_up2], axis=3))

        return out


class ASB(Model):
    def __init__(self, filters, block_name, inputs_size, groups=2):
        super(ASB, self).__init__()
        self.filters = filters
        self.block_name = block_name + '_'
        self.groups = groups
        self.inputs_size = inputs_size

        self.cbr1 = Con_Bn_Act(filters=self.filters//2, kernel_size=(1, 1), name=self.block_name+'ASB_cbr1x1_1')
        self.dcbr1 = DW_Con_Bn_Act(filters=self.filters//2, name='ASB_dcbr1')
        self.cbr2 = Con_Bn_Act(filters=self.filters//2, kernel_size=(1, 1), name=self.block_name+'ASB_cbr1x1_2')

        # self.gap = GlobalAveragePooling2D(keepdims=True)
        self.cbr3 = Con_Bn_Act(filters=self.filters//2, kernel_size=(1, 1), activation='sigmoid',
                               name=self.block_name+'ASB_cbr1x1_3')

    def call(self, inputs, training=None, mask=None):
        shortcut, part_inputs = tf.split(inputs, 2, axis=3)

        cbr1 = self.cbr1(part_inputs)
        dcbr1 = self.dcbr1(cbr1)
        cbr2 = self.cbr2(dcbr1)

        gap = tf.reduce_mean(cbr2, [1, 2], keepdims=True)
        cbr3 = self.cbr3(gap)

        mul = multiply([cbr2, cbr3])

        out = tf.concat([shortcut, mul], axis=3)
        out = channel_shuffle(out, self.inputs_size, 2)

        return out
