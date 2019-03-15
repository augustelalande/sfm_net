import os

import tensorflow as tf
import tensorflow.contrib.summary as summary
from tensorboard.plugins.beholder import Beholder

from sfm import SfMNet
from data_reader import DataReader
from utils import *
from loss_utils import *


S_max = int(1e5)
batch_size = 16
lr = 1e-5

logs_path = "/localdata/auguste/logs_sfm"


if __name__ == '__main__':
    session_name = get_session_name()
    session_logs_path = os.path.join(logs_path, session_name)

    global_step = tf.train.get_or_create_global_step()

    data_reader = DataReader(
        "stereo", batch_size, "/localdata/auguste/kitti-raw")
    model = SfMNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    beholder = Beholder(logs_path)
    writer = summary.create_file_writer(session_logs_path, max_queue=0)
    writer.set_as_default()

    with summary.record_summaries_every_n_global_steps(50):

        # Train

        f0, f1 = data_reader.read()
        depth, points, flow, obj_masks, pc_t = model(f0, f1)
        depth1, *_ = model(f1, f0)

        f_loss, f1_t = frame_loss(f0, f1, points)
        fb_loss = forward_backward_consistency_loss(
            depth, depth1, points, pc_t)

        loss = f_loss
        # + \
        #     spatial_smoothness_loss(depth) + \
        #     spatial_smoothness_loss(flow) + \
        #     forward_backward_consistency_loss()
        optimize = optimizer.minimize(loss, global_step=global_step)

        summary.scalar("loss", loss, family="train")

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

        for s in range(S_max):
            l, *_ = sess.run(
                [loss, optimize, summary.all_summary_ops()])
            beholder.update(session=sess)

            if s % 50 == 0:
                print("Iteration: {}  Loss: {}".format(s, l))
