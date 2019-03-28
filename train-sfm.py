import os

import tensorflow as tf
import tensorflow.contrib.summary as summary
from tensorboard.plugins.beholder import Beholder

from sfm import SfMNet
from data_reader import DataReader
from utils import *
from loss_utils import *


S_max = int(1e5)
batch_size = 10
lr = 1e-4

logs_path = "/localdata/auguste/logs_sfm"
models_path = "/localdata/auguste/models"


if __name__ == '__main__':
    session_name = get_session_name()
    session_logs_path = os.path.join(logs_path, session_name)

    global_step = tf.train.get_or_create_global_step()

    data_reader = DataReader(
        "sequence", batch_size, "/localdata/auguste/kitti-raw")
    model = SfMNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    beholder = Beholder(logs_path)
    writer = summary.create_file_writer(session_logs_path, max_queue=0)
    writer.set_as_default()

    with summary.record_summaries_every_n_global_steps(50):

        # Train

        f0, f1 = data_reader.read()
        depth, points, flow, obj_p, cam_p, pc_t, motion_maps = model(f0, f1)
        depth1, points1, flow1, _, _, pc_t1, motion_maps1 = model(f1, f0)

        f_loss, f1_t = frame_loss(f0, f1, points)
        f_loss1, _ = frame_loss(f1, f0, points1)

        fb_loss = forward_backward_consistency_loss(depth1, points, pc_t)
        fb_loss1 = forward_backward_consistency_loss(depth, points1, pc_t1)

        ss_loss_d = spatial_smoothness_loss(depth / 100, order=2)
        ss_loss_d1 = spatial_smoothness_loss(depth1 / 100, order=2)

        ss_loss_f = spatial_smoothness_loss(flow, order=2)
        ss_loss_f1 = spatial_smoothness_loss(flow1, order=2)

        b, h, w, k, c = motion_maps.shape
        ss_loss_m = spatial_smoothness_loss(
            tf.reshape(motion_maps, [b, h, w, k * c]))
        ss_loss_m1 = spatial_smoothness_loss(
            tf.reshape(motion_maps1, [b, h, w, k * c]))

        ss_loss = ss_loss_d + ss_loss_d1 + ss_loss_f + \
            ss_loss_f1 + ss_loss_m + ss_loss_m1

        loss = f_loss + f_loss1 + fb_loss + fb_loss1 + ss_loss
        optimize = optimizer.minimize(loss, global_step=global_step)

        summary.scalar("loss", loss, family="train")
        summary.scalar("frame loss forward", f_loss, family="train")
        summary.scalar("frame loss backward", f_loss1, family="train")
        summary.scalar("fb loss forward", fb_loss, family="train")
        summary.scalar("fb loss backward", fb_loss1, family="train")
        summary.scalar("ss loss", ss_loss, family="train")

        summary.histogram("depth_hist", depth)
        summary.histogram("obj masks", obj_masks)
        summary.histogram("flow_x_hist", flow[:, :, :, 0], family="flow")
        summary.histogram("flow_y_hist", flow[:, :, :, 1], family="flow")

        summary.image("frame0", cast_im(f0), max_images=3)
        summary.image("frame1", cast_im(f1), max_images=3)
        summary.image("frame1_t", cast_im(f1_t), max_images=3)
        summary.image("depth", cast_depth(depth), max_images=3)
        summary.image("optical_flow", cast_flow(flow), max_images=3)
        summary.image("object masks", cast_im(obj_masks), max_images=3)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        summary.initialize(graph=tf.get_default_graph())

        model.load_weights(os.path.join(models_path, "sfm.h5"))

        for s in range(S_max):
            l, *_ = sess.run(
                [loss, optimize, summary.all_summary_ops()])
            beholder.update(session=sess)

            if s % 50 == 0:
                print("Iteration: {}  Loss: {}".format(s, l))

            if s % 5000 == 0 and not s == 0:
                model.save_weights(os.path.join(models_path, "sfm.h5"))
