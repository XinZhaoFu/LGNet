from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D, \
    DepthwiseConv2D, UpSampling2D, MaxPooling2D, concatenate
from tensorflow.keras import Model, regularizers, Sequential
import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, MaxPooling2D, \
    concatenate


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


class Con_Unit_Utils(Model):
    def __init__(self, filters, init_input_shape, activation='relu'):
        super(Con_Unit_Utils, self).__init__()
        self.filters = filters
        self.init_input_shape = init_input_shape
        width, _, _ = init_input_shape
        self.follow_input_shape = (width, width, self.filters)
        self.activation = activation

        self.con = Conv2D(filters=self.filters, kernel_size=(3, 3), input_shape=self.init_input_shape, padding='same', use_bias=False, kernel_regularizer=regularizers.l2())
        self.bn = BatchNormalization(input_shape=self.follow_input_shape)
        self.act = Activation(self.activation)

    def call(self, inputs, training=None, mask=None):
        con = self.con(inputs)
        bn = self.bn(con)
        out = self.act(bn)
        return out


class Con_Block_Utils(Model):
    def __init__(self, filters, input_width, input_channel, num_con_unit=1):
        super(Con_Block_Utils, self).__init__()
        self.filters = filters
        self.input_width = input_width
        self.input_channel = input_channel
        self.num_con_unit = num_con_unit
        self.init_input_shape = (self.input_width, self.input_width, self.input_channel)
        self.follow_input_shape = (self.input_width, self.input_width, self.filters)

        self.con_blocks = Sequential()
        for id_unit in range(self.num_con_unit):
            if id_unit == 0:
                block = Con_Unit_Utils(filters=self.filters, init_input_shape=self.init_input_shape)
            else:
                block = Con_Unit_Utils(filters=self.filters, init_input_shape=self.follow_input_shape)
            self.con_blocks.add(block)

    def call(self, inputs, training=None, mask=None):
        out = self.con_blocks(inputs)
        return out


class Up_Block_Utils(Model):
    def __init__(self, filters, input_width, input_channel, num_con_unit=1):
        super().__init__()
        self.filters = filters
        self.input_width = input_width
        self.input_channel = input_channel
        self.num_con_unit = num_con_unit
        self.init_input_shape = (self.input_width, self.input_width, self.input_channel)
        self.follow_input_shape = (self.input_width, self.input_width, self.filters)

        self.con_blocks = Sequential()
        for id_unit in range(self.num_con_unit):
            if id_unit == 0:
                block = Con_Unit_Utils(filters=self.filters, init_input_shape=self.init_input_shape)
            else:
                block = Con_Unit_Utils(filters=self.filters, init_input_shape=self.follow_input_shape)
            self.con_blocks.add(block)
        self.up = UpSampling2D()

    def call(self, inputs, training=None, mask=None):
        con = self.con_blocks(inputs)
        out = self.up(con)
        return out


class LGNet_Utils(Model):
    def __init__(self, filters=32, img_width=512, num_class=2, num_con_unit=1):
        super(LGNet_Utils, self).__init__()
        self.filters = filters
        self.input_width = img_width
        self.num_class = num_class
        self.num_con_unit = num_con_unit

        self.con_block1 = Con_Block_Utils(filters=self.filters, input_width=self.input_width, input_channel=1, num_con_unit=self.num_con_unit)
        self.con_block2 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 2, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block3 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 4, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block4 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 8, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block5 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 16, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block6 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 32, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_block7 = Con_Block_Utils(filters=self.filters, input_width=self.input_width / 64, input_channel=self.filters, num_con_unit=self.num_con_unit)

        self.con_up7 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 64, input_channel=self.filters, num_con_unit=self.num_con_unit)
        self.con_up6 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 32, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)
        self.con_up5 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 16, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)
        self.con_up4 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 8, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)
        self.con_up3 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 4, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)
        self.con_up2 = Up_Block_Utils(filters=self.filters, input_width=self.input_width / 2, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)
        self.con_up1 = Con_Block_Utils(filters=self.filters, input_width=self.input_width, input_channel=self.filters * 2, num_con_unit=self.num_con_unit)

        self.con_end = Con_Unit_Utils(filters=self.num_class, init_input_shape=(self.input_width, self.input_width, self.filters), activation='softmax')

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs, training=None, mask=None):
        con1 = self.con_block1(inputs)

        pool2 = self.pool(con1)
        con2 = self.con_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.con_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.con_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.con_block5(pool5)

        pool6 = self.pool(con5)
        con6 = self.con_block6(pool6)

        pool7 = self.pool(con6)
        con7 = self.con_block7(pool7)

        up7 = self.con_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.con_up6(merge6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.con_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.con_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.con_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.con_up2(merge2)

        merge1 = concatenate([up2, con1], axis=3)
        up1 = self.con_up1(merge1)

        out = self.con_end(up1)

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
