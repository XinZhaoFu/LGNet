from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, \
    DepthwiseConv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import UpSampling2D
import tensorflow as tf


class Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 dilation_rate=1,
                 name=None,
                 kernel_regularizer=False,
                 train_able=True):
        """
        Conv2D + BN + activation

        :param filters:
        :param kernel_size:
        :param padding:
        :param strides:
        :param activation:
        :param dilation_rate:
        :param name:
        :param kernel_regularizer:
        :param train_able:
        """
        super(Con_Bn_Act, self).__init__()
        self.kernel_regularizer = kernel_regularizer
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.block_name = name
        self.train_able = train_able

        if self.kernel_regularizer:
            self.con_regularizer = regularizers.l2()
        else:
            self.con_regularizer = None

        # kernel_initializer_special_cases = ['glorot_uniform',
        # 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']
        self.con = Conv2D(filters=self.filters,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          strides=self.strides,
                          use_bias=False,
                          dilation_rate=(self.dilation_rate, self.dilation_rate),
                          name=self.block_name,
                          kernel_regularizer=self.con_regularizer,
                          kernel_initializer='glorot_uniform')
        if self.train_able is False:
            self.con.trainable = False
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)

    def call(self, inputs, training=None, mask=None):
        con = self.con(inputs)
        bn = self.bn(con)
        out = self.act(bn)

        return out


class Sep_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 strides=1,
                 activation='relu',
                 name=None):
        super(Sep_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.block_name = name

        self.con = SeparableConv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   strides=self.strides,
                                   use_bias=False,
                                   name=self.block_name)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)

    def call(self, inputs, training=None, mask=None):
        con = self.con(inputs)
        bn = self.bn(con)
        out = self.act(bn)

        return out


class DW_Con_Bn_Act(Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=1,
                 use_bias=False,
                 padding='same',
                 name=None,
                 activation='relu'):
        super(DW_Con_Bn_Act, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.padding = padding
        self.block_name = name
        self.activation = activation

        self.dw_con = DepthwiseConv2D(kernel_size=self.kernel_size,
                                      strides=self.strides,
                                      use_bias=self.use_bias,
                                      padding=self.padding,
                                      name=self.block_name)
        self.bn = BatchNormalization()
        self.act = Activation(self.activation)

    def call(self, inputs, training=None, mask=None):
        con = self.dw_con(inputs)
        bn = self.bn(con)
        out = self.act(bn)

        return out


class Aspp(Model):
    def __init__(self, filters=256, dila_rate1=6, dila_rate2=12, dila_rate3=18, dila_rate4=24):
        super(Aspp, self).__init__()
        self.filters = filters

        self.con1x1 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), activation='relu', name='aspp_con1x1')

        self.dila_con1 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate1, name='aspp_dila_con1')
        self.dila_con2 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate2, name='aspp_dila_con2')
        self.dila_con3 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate3, name='aspp_dila_con3')
        self.dila_con4 = Con_Bn_Act(filters=self.filters, dilation_rate=dila_rate4, name='aspp_dila_con3')

        self.pooling_1 = MaxPooling2D(name='aspp_pooling_pooling', padding='same')
        self.pooling_2 = Conv2D(filters=self.filters, kernel_size=(1, 1), name='aspp_pooling_con1x1')
        self.pooling_3 = UpSampling2D(name='aspp_pooling_upsampling')

        self.concat_2 = Con_Bn_Act(filters=self.filters, kernel_size=(1, 1), name='aspp_concate_con1x1')

    def call(self, inputs, training=None, mask=None):
        con1x1 = self.con1x1(inputs)

        dila_con6x6 = self.dila_con1(inputs)
        dila_con12x12 = self.dila_con2(inputs)
        dila_con18x18 = self.dila_con3(inputs)

        pooling_1 = self.pooling_1(inputs)
        pooling_2 = self.pooling_2(pooling_1)
        pooling_3 = self.pooling_3(pooling_2)

        concat_1 = concatenate([con1x1, dila_con6x6, dila_con12x12, dila_con18x18, pooling_3], axis=3)
        out = self.concat_2(concat_1)

        return out


class CBR_Block(Model):
    def __init__(self, filters, num_cbr=1, block_name=None):
        super(CBR_Block, self).__init__()
        self.filters = filters
        self.num_cbr = num_cbr
        self.block_name = None
        if block_name is not None and type(block_name) == str:
            self.block_name = block_name

        self.con_blocks = Sequential()
        for index in range(self.num_cbr):
            if self.block_name is not None:
                block = Con_Bn_Act(filters=self.filters, name=self.block_name + '_Con_Block_' + str(index+1))
            else:
                block = Con_Bn_Act(filters=self.filters, name='Con_Block_' + str(index + 1))
            self.con_blocks.add(block)

    def call(self, inputs, training=None, mask=None):
        out = self.con_blocks(inputs)

        return out


class SCBR_Block(Model):
    def __init__(self, filters, num_scbr=1, block_name=None):
        super(SCBR_Block, self).__init__()
        self.filters = filters
        self.num_scbr = num_scbr
        self.block_name = None
        if block_name is not None and type(block_name) == str:
            self.block_name = block_name

        self.con_blocks = Sequential()
        for index in range(self.num_scbr):
            if self.block_name is not None:
                block = Sep_Con_Bn_Act(filters=self.filters, name=self.block_name + '_Con_Block_' + str(index+1))
            else:
                block = Sep_Con_Bn_Act(filters=self.filters, name='Con_Block_' + str(index + 1))
            self.con_blocks.add(block)

    def call(self, inputs, training=None, mask=None):
        out = self.con_blocks(inputs)

        return out


class Up_CBR_Block(Model):
    def __init__(self, filters, num_cbr=1, block_name=''):
        super(Up_CBR_Block, self).__init__()
        self.filters = filters
        self.num_cbr = num_cbr
        self.block_name = None
        self.block_name = block_name

        self.con_blocks = CBR_Block(filters=self.filters, num_cbr=self.num_cbr, block_name=self.block_name)
        self.up = UpSampling2D(name=self.block_name + '_up_sampling')

    def call(self, inputs, training=None, mask=None):
        con = self.con_blocks(inputs)
        out = self.up(con)
        return out


def channel_shuffle(inputs, inputs_size, group=2):
    """
    用于通道混合

    :param inputs_size:
    :param inputs:
    :param group:
    :return:
    """
    in_shape = inputs.shape.as_list()
    in_channel = in_shape[3]

    assert in_channel % group == 0
    out = tf.reshape(inputs, [-1, inputs_size, inputs_size, in_channel // group, group])
    out = tf.transpose(out, [0, 1, 2, 4, 3])
    out = tf.reshape(out, [-1, inputs_size, inputs_size, in_channel])

    return out
