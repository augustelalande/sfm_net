import tensorflow as tf
from tensorflow.losses import mean_squared_error as mse


def frame_loss(x0, x1, points):
    warp = points_to_warp(points)
    x1_t = tf.contrib.resampler.resampler(x1, warp)
    return mse(x0, x1_t), x1_t


def spatial_smoothness_loss(x, order=1):
    b, h, w, c = x.shape
    gradients = tf.image.sobel_edges(x)
    for i in range(order - 1):
        gradients = tf.reshape(gradients, [b, h, w, -1])
        gradients = tf.image.sobel_edges(gradients)
    return tf.reduce_mean(tf.square(gradients))


def forward_backward_consistency_loss(d1, points, pc_t):
    warp = points_to_warp(points)
    d1_t = tf.contrib.resampler.resampler(d1, warp)
    Z0 = pc_t[:, :, :, 2:3]
    return mse(d1_t / 100, Z0 / 100)


def points_to_warp(points):
    b, h, w, c = points.shape
    warp_x = points[:, :, :, 0]
    warp_y = points[:, :, :, 1]

    warp_x = warp_x * tf.cast(w - 1, tf.float32)
    warp_y = warp_y * tf.cast(h - 1, tf.float32)
    return tf.stack([warp_x, warp_y], -1)
