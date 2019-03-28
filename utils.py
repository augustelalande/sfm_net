import tensorflow as tf
import tensorflow.contrib.summary as summary
import time


def cast_im(x):
    return tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)


def cast_depth(x):
    return cast_im(x / tf.reduce_max(x))


def cast_flow(flow):
    x = flow[:, :, :, 0]
    y = flow[:, :, :, 1]
    hue = tf.atan2(y, x)
    mag = tf.sqrt(x ** 2 + y ** 2)
    value = tf.minimum(mag, 0.1) / 0.1
    saturation = tf.ones_like(value)
    hsv_im = tf.stack([hue, saturation, value], -1)
    rgb_im = tf.image.hsv_to_rgb(hsv_im)
    return cast_im(rgb_im)


def summary_text(name, tensor, family=None, step=None):
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    return summary.generic(
        name, tensor, metadata=meta, family=family, step=step)


def get_session_name():
    current_time = int(time.time())
    description = input("Describe the session: ")
    session_name = "s{}: {}".format(current_time, description)
    print(session_name)
    return session_name
