import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.activations import relu, sigmoid, tanh, softmax

from conv_deconv_net import ConvDeconvNet


class MotionNet(tf.keras.Model):
    def __init__(self, num_masks=3):
        super().__init__()
        self.num_masks = num_masks

        self.cd_net = ConvDeconvNet()
        self.obj_mask = Conv2D(num_masks, 1)

        self.d1 = Dense(512, activation=relu)
        self.d2 = Dense(512, activation=relu)

        self.cam_t = Dense(3)
        self.cam_p = Dense(600, activation=softmax)
        self.cam_r = Dense(3, activation=tanh)

        self.obj_t = Dense(3 * num_masks)
        self.obj_p = Dense(600 * num_masks)
        self.obj_r = Dense(3 * num_masks, activation=tanh)

    def call(self, f0, f1, sharpness_multiplier):
        x = tf.concat([f0, f1], -1)
        x, r = self.cd_net(x)
        b, *_ = r.shape

        r = tf.reshape(r, [b, -1])
        r = self.d1(r)
        r = self.d2(r)

        obj_mask = sigmoid(self.obj_mask(x) * sharpness_multiplier)

        obj_t = self.obj_t(r)
        obj_p = self.obj_p(r)
        obj_p = tf.reshape(obj_p, [-1, self.num_masks, 600])
        obj_p = softmax(obj_p)
        obj_r = self.obj_r(r)

        cam_t = self.cam_t(r)
        cam_p = self.cam_p(r)
        cam_r = self.cam_r(r)

        return (obj_mask, obj_t, obj_p, obj_r), (cam_t, cam_p, cam_r)
