import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.activations import relu


class BConv2D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = Conv2D(*args, **kwargs)
        # self.batch_norm = BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        return relu(x)


class BConv2DTranspose(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = Conv2DTranspose(*args, **kwargs)
        # self.batch_norm = BatchNormalization()

    def call(self, x):
        x = self.conv_transpose(x)
        # x = self.batch_norm(x)
        return relu(x)


class ConvDeconvNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        kwargs = {'padding': 'SAME'}

        self.c11 = BConv2D(32, 3, **kwargs)

        self.c21 = BConv2D(64, 3, strides=2, **kwargs)
        self.c22 = BConv2D(64, 3, **kwargs)

        self.c31 = BConv2D(128, 3, strides=2, **kwargs)
        self.c32 = BConv2D(128, 3, **kwargs)

        self.c41 = BConv2D(256, 3, strides=2, **kwargs)
        self.c42 = BConv2D(256, 3, **kwargs)

        self.c51 = BConv2D(512, 3, strides=2, **kwargs)
        self.c52 = BConv2D(512, 3, **kwargs)

        self.c61 = BConv2D(1024, 3, strides=2, **kwargs)
        self.c62 = BConv2D(1024, 3, **kwargs)

        kwargs = {'strides': 2, 'padding': 'SAME'}

        self.u5 = BConv2DTranspose(512, 3, **kwargs)
        self.u4 = BConv2DTranspose(256, 3, **kwargs)
        self.u3 = BConv2DTranspose(128, 3, **kwargs)
        self.u2 = BConv2DTranspose(64, 3, **kwargs)
        self.u1 = BConv2DTranspose(32, 3, **kwargs)

    def call(self, x):
        x1 = self.c11(x)

        x2 = self.c21(x1)
        x2 = self.c22(x2)

        x3 = self.c31(x2)
        x3 = self.c32(x3)

        x4 = self.c41(x3)
        x4 = self.c42(x4)

        x5 = self.c51(x4)
        x5 = self.c52(x5)

        x6 = self.c61(x5)
        embedding = self.c62(x6)

        u5 = self.u5(embedding)
        u5 = tf.concat([x5, u5], -1)

        u4 = self.u4(u5)
        u4 = tf.concat([x4, u4], -1)

        u3 = self.u3(u4)
        u3 = tf.concat([x3, u3], -1)

        u2 = self.u2(u3)
        u2 = tf.concat([x2, u2], -1)

        u1 = self.u1(u2)
        u1 = tf.concat([x1, u1], -1)
        return u1, embedding
