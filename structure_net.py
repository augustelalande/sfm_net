import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from conv_deconv_net import ConvDeconvNet


class StructureNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.cd_net = ConvDeconvNet()
        self.depth = Conv2D(1, 1)

    def call(self, x):
        x, _ = self.cd_net(x)
        x = self.depth(x)
        depth = tf.clip_by_value(x, 1, 100)
        pc = depth_to_point(depth)
        return depth, pc


def depth_to_point(depth, camera_intrinsics=(0.5, 0.5, 1.0)):
    cx, cy, cf = camera_intrinsics
    b, h, w, c = depth.shape

    x_l = tf.linspace(-cx, 1 - cx, w)
    y_l = tf.linspace(-cy, 1 - cy, h)

    x, y = tf.meshgrid(x_l, y_l)
    x, y = x / cf, y / cf
    f = tf.ones_like(x)

    grid = tf.stack([x, y, f], -1)
    return depth * grid
