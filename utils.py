import tensorflow as tf
import tensorflow.contrib.summary as summary
import time
import os
from git import Repo


from sfm import _pivot_point


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


def obj_summary(obj_params):
    obj_mask, obj_t, obj_p, obj_r = obj_params

    template = (
        "- | im 0; 1 | im 0; 2 | im 0; 3 | "
        "im 1; 1 | im 1; 2 | im 1; 3 | "
        "im 2; 1 | im 2; 2 | im 2; 3\n"
        "---|---|---|---|---|---|---|---|---|---\n"
        "trans x | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "trans y | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "trans z | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "-|-|-|-|-|-|-|-|-|-\n"
        "rot x | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "rot y | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "rot z | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "-|-|-|-|-|-|-|-|-|-\n"
        "pivot x | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "pivot y | {} | {} | {} | {} | {} | {} | {} | {} | {}\n"
        "pivot z | {} | {} | {} | {} | {} | {} | {} | {} | {}"
    )

    t = tf.reshape(obj_t, [-1, 3, 3])
    r = tf.reshape(obj_r, [-1, 3, 3]) * 180
    p = tf.reshape(_pivot_point(obj_p), [-1, 3, 3])
    tensor = tf.strings.format(
        template,
        [
            t[0, 0, 0], t[0, 1, 0], t[0, 2, 0],
            t[1, 0, 0], t[1, 1, 0], t[1, 2, 0],
            t[2, 0, 0], t[2, 1, 0], t[2, 2, 0],
            t[0, 0, 1], t[0, 1, 1], t[0, 2, 1],
            t[1, 0, 1], t[1, 1, 1], t[1, 2, 1],
            t[2, 0, 1], t[2, 1, 1], t[2, 2, 1],
            t[0, 0, 2], t[0, 1, 2], t[0, 2, 2],
            t[1, 0, 2], t[1, 1, 2], t[1, 2, 2],
            t[2, 0, 2], t[2, 1, 2], t[2, 2, 2],

            r[0, 0, 0], r[0, 1, 0], r[0, 2, 0],
            r[1, 0, 0], r[1, 1, 0], r[1, 2, 0],
            r[2, 0, 0], r[2, 1, 0], r[2, 2, 0],
            r[0, 0, 1], r[0, 1, 1], r[0, 2, 1],
            r[1, 0, 1], r[1, 1, 1], r[1, 2, 1],
            r[2, 0, 1], r[2, 1, 1], r[2, 2, 1],
            r[0, 0, 2], r[0, 1, 2], r[0, 2, 2],
            r[1, 0, 2], r[1, 1, 2], r[1, 2, 2],
            r[2, 0, 2], r[2, 1, 2], r[2, 2, 2],

            p[0, 0, 0], p[0, 1, 0], p[0, 2, 0],
            p[1, 0, 0], p[1, 1, 0], p[1, 2, 0],
            p[2, 0, 0], p[2, 1, 0], p[2, 2, 0],
            p[0, 0, 1], p[0, 1, 1], p[0, 2, 1],
            p[1, 0, 1], p[1, 1, 1], p[1, 2, 1],
            p[2, 0, 1], p[2, 1, 1], p[2, 2, 1],
            p[0, 0, 2], p[0, 1, 2], p[0, 2, 2],
            p[1, 0, 2], p[1, 1, 2], p[1, 2, 2],
            p[2, 0, 2], p[2, 1, 2], p[2, 2, 2]
        ]
    )

    pivot_heatmap = tf.reshape(obj_p, [-1, 20, 30, 1])
    summary.image("pivot heatmap obj", cast_im(pivot_heatmap), max_images=9)

    return summary_text("obj summary", tensor)


def cam_summary(cam_params):
    cam_t, cam_p, cam_r = cam_params

    template = (
        "- | im 0 | im 1 | im 2\n"
        "---|---|---|---\n"
        "trans x | {} | {} | {}\n"
        "trans y | {} | {} | {}\n"
        "trans z | {} | {} | {}\n"
        "-|-|-|-\n"
        "rot x | {} | {} | {}\n"
        "rot y | {} | {} | {}\n"
        "rot z | {} | {} | {}\n"
        "-|-|-|-\n"
        "pivot x | {} | {} | {}\n"
        "pivot y | {} | {} | {}\n"
        "pivot z | {} | {} | {}"
    )

    t = cam_t
    r = cam_r * 180
    p = tf.reshape(_pivot_point(cam_p), [-1, 3])
    tensor = tf.strings.format(
        template,
        [
            t[0, 0], t[1, 0], t[2, 0],
            t[0, 1], t[1, 1], t[2, 1],
            t[0, 2], t[1, 2], t[2, 2],

            r[0, 0], r[1, 0], r[2, 0],
            r[0, 1], r[1, 1], r[2, 1],
            r[0, 2], r[1, 2], r[2, 2],

            p[0, 0], p[1, 0], p[2, 0],
            p[0, 1], p[1, 1], p[2, 1],
            p[0, 2], p[1, 2], p[2, 2]
        ]
    )

    pivot_heatmap = tf.reshape(cam_p, [-1, 20, 30, 1])
    summary.image("pivot heatmap cam", cast_im(pivot_heatmap), max_images=3)

    return summary_text("cam summary", tensor)


def summary_text(name, tensor, family=None, step=None):
    meta = tf.SummaryMetadata()
    meta.plugin_data.plugin_name = "text"
    return summary.generic(
        name, tensor, metadata=meta, family=family, step=step)


def sharpness_multiplier(init_val, step, num_steps, end_val):
    step = tf.cast(step, tf.float32)
    val = init_val + (end_val - init_val) * (step / num_steps) ** 2
    return tf.minimum(val, end_val)


def get_session_name():
    current_time = int(time.time())
    description = input("Describe the session: ")
    session_name = "s{}: {}".format(current_time, description)
    print(session_name)
    return session_name


def commit_changes(session_name):
    path, _ = os.path.split(__file__)
    repo = Repo(path)
    assert not repo.bare
    assert str(repo.active_branch) == "dev"
    index = repo.index
    index.add(["*.py"])
    index.commit(session_name)
