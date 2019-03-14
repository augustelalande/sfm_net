import tensorflow as tf
import math

from structure_net import StructureNet
from motion_net import MotionNet


class SfMNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.structure = StructureNet()
        self.motion = MotionNet()

    def call(self, f0, f1):
        depth, pc = self.structure(f0)
        obj_params, cam_params = self.motion(f0, f1)
        pc_t = apply_obj_transform(pc, *obj_params)
        pc_t = apply_cam_transform(pc, *cam_params)
        points, flow = optical_flow(pc_t)
        return depth, points, flow


def apply_obj_transform(pc, obj_mask, obj_t, obj_p, obj_r, num_masks=3):
    b, h, w, c = pc.shape

    p = _point_prior(obj_p)
    R = _r_mat(tf.reshape(obj_r, [b, num_masks, 3]))

    p = tf.reshape(p, [-1, 1, 1, num_masks, 3])
    t = tf.reshape(obj_t, [-1, 1, 1, num_masks, 3])
    R = tf.reshape(R, [-1, 1, 1, num_masks, 3, 3])
    R = tf.tile(R, [1, h, w, 1, 1, 1])

    pc = tf.reshape(pc, [b, h, w, 1, 3])
    mask = tf.reshape(obj_mask, [b, h, w, num_masks, 1])

    pc_t = pc - p
    pc_t = tf.expand_dims(pc_t, -1)
    pc_t = R @ pc_t
    pc_t = tf.reshape(pc_t, [b, h, w, num_masks, 3])
    pc_t = pc_t + t - pc

    pc = tf.reshape(pc, [b, h, w, 3])
    pc_t = pc + tf.reduce_sum(mask * pc_t, -2)
    return pc_t


def apply_cam_transform(pc, cam_t, cam_p, cam_r):
    b, h, w, c = pc.shape

    p = _point_prior(obj_p)
    R = _r_mat(tf.reshape(obj_r, [b, num_masks, 3]))

    p = tf.reshape(p, [-1, 1, 1, 3])
    t = tf.reshape(obj_t, [-1, 1, 1, 3])
    R = tf.reshape(R, [-1, 1, 1, 3, 3])
    R = tf.tile(R, [1, h, w, 1, 1])

    pc_t = pc - p
    pc_t = tf.expand_dims(pc_t, -1)
    pc_t = R @ pc_t
    pc_t = tf.reshape(pc_t, [b, h, w, 3])
    pc_t = pc_t + t
    return pc_t


def optical_flow(pc, camera_intrinsics=(0.5, 0.5, 1.0)):
    points = _project_2d(pc, camera_intrinsics)
    b, h, w, c = points.shape

    x_l = tf.linspace(0.0, 1.0, w)
    y_l = tf.linspace(0.0, 1.0, h)
    x, y = tf.meshgrid(x_l, y_l)
    pos = tf.stack([x, y], -1)
    flow = points - pos
    return points, flow


def _project_2d(pc, camera_intrinsics):
    cx, cy, cf = camera_intrinsics

    X = pc[:, :, :, 0]
    Y = pc[:, :, :, 1]
    Z = pc[:, :, :, 2]

    x = cf * X / Z + cx
    y = cf * Y / Z + cy
    return tf.stack([x, y], -1)


def _point_prior(p):
    p = tf.reshape(p, [-1, 30, 20])
    p_x = tf.reduce_sum(p, 2)
    p_y = tf.reduce_sum(p, 1)

    x_l = tf.linspace(-30.0, 30.0, 30)
    y_l = tf.linspace(-20.0, 20.0, 20)

    P_x = tf.reduce_mean(p_x * x_l, -1)
    P_y = tf.reduce_mean(p_y * y_l, -1)
    ground = tf.ones_like(P_x)

    P = tf.stack([P_x, P_y, ground], 1)
    return P


def _r_mat(r):
    alpha = r[:, :, 0] * math.pi
    beta = r[:, :, 1] * math.pi
    gamma = r[:, :, 2] * math.pi

    zero = tf.zeros_like(alpha)
    one = tf.ones_like(alpha)

    R_x = tf.stack([
        tf.stack([tf.cos(alpha), -tf.sin(alpha), zero], -1),
        tf.stack([tf.sin(alpha), tf.cos(alpha), zero], -1),
        tf.stack([zero, zero, one], -1),
    ], -2)

    R_y = tf.stack([
        tf.stack([tf.cos(beta), zero, tf.sin(beta)], -1),
        tf.stack([zero, one, zero], -1),
        tf.stack([-tf.sin(beta), zero, tf.cos(beta)], -1),
    ], -2)

    R_z = tf.stack([
        tf.stack([one, zero, zero], -1),
        tf.stack([zero, tf.cos(gamma), -tf.sin(gamma)], -1),
        tf.stack([zero, tf.sin(gamma), tf.cos(gamma)], -1),
    ], -2)

    return R_x @ R_y @ R_z


if __name__ == '__main__':
    tf.enable_eager_execution()

    pc = tf.ones([10, 32, 32, 3])
    obj_mask = tf.ones([10, 32, 32, 5])
    obj_t = tf.ones([10, 5 * 3])
    obj_p = tf.ones([10, 5 * 600])
    obj_r = tf.ones([10, 5 * 3])

    optical_flow(pc)
