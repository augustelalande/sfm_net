import tensorflow as tf
import time


def cast_im(x):
    return tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)


def cast_depth(x):
    return tf.cast(tf.clip_by_value(x * 255 / 100, 0, 255), tf.uint8)


def get_session_name():
    current_time = int(time.time())
    description = input("Describe the session: ")
    session_name = "s{}: {}".format(current_time, description)
    print(session_name)
    return session_name
