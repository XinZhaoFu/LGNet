from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, concatenate
from model.utils import Con_Bn_Act, Up_CBR_Block


class UNetM(Model):
    def __init__(self, filters=32, num_class=3, end_activation='softmax'):
        super(UNetM, self).__init__()
        self.filters = filters
        self.num_class = num_class
        self.end_activation = end_activation

        self.cbr_block1 = Con_Bn_Act(filters=self.filters, name='down1')
        self.cbr_block2 = Con_Bn_Act(filters=self.filters, name='down2')
        self.cbr_block3 = Con_Bn_Act(filters=self.filters, name='down3')
        self.cbr_block4 = Con_Bn_Act(filters=self.filters, name='down4')
        self.cbr_block5 = Con_Bn_Act(filters=self.filters, name='down5')
        self.cbr_block6 = Con_Bn_Act(filters=self.filters, name='down6')
        self.cbr_block7 = Con_Bn_Act(filters=self.filters, name='down7')

        self.cbr_block_up7 = Up_CBR_Block(filters=self.filters, block_name='up7')
        self.cbr_block_up6 = Up_CBR_Block(filters=self.filters, block_name='up6')
        self.cbr_block_up5 = Up_CBR_Block(filters=self.filters, block_name='up5')
        self.cbr_block_up4 = Up_CBR_Block(filters=self.filters, block_name='up4')
        self.cbr_block_up3 = Up_CBR_Block(filters=self.filters, block_name='up3')
        self.cbr_block_up2 = Up_CBR_Block(filters=self.filters, block_name='up2')
        self.cbr_block_up1 = Con_Bn_Act(filters=self.filters, name='up1')

        self.con_end = Con_Bn_Act(filters=self.num_class, activation=self.end_activation)

        self.pool = MaxPooling2D(padding='same')

    def call(self, inputs):
        con1 = self.cbr_block1(inputs)

        pool2 = self.pool(con1)
        con2 = self.cbr_block2(pool2)

        pool3 = self.pool(con2)
        con3 = self.cbr_block3(pool3)

        pool4 = self.pool(con3)
        con4 = self.cbr_block4(pool4)

        pool5 = self.pool(con4)
        con5 = self.cbr_block5(pool5)

        pool6 = self.pool(con5)
        con6 = self.cbr_block6(pool6)

        pool7 = self.pool(con6)
        con7 = self.cbr_block7(pool7)

        up7 = self.cbr_block_up7(con7)

        merge6 = concatenate([up7, con6], axis=3)
        up6 = self.cbr_block_up6(merge6)

        merge5 = concatenate([up6, con5], axis=3)
        up5 = self.cbr_block_up5(merge5)

        merge4 = concatenate([up5, con4], axis=3)
        up4 = self.cbr_block_up4(merge4)

        merge3 = concatenate([up4, con3], axis=3)
        up3 = self.cbr_block_up3(merge3)

        merge2 = concatenate([up3, con2], axis=3)
        up2 = self.cbr_block_up2(merge2)

        merge1 = concatenate([up2, con1], axis=3)
        up1 = self.cbr_block_up1(merge1)

        out = self.con_end(up1)

        return out
