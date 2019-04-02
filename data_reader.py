import tensorflow as tf
import collections
import os

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'frame_height', 'frame_width', 'num_channels']
)

_DATASETS = dict(
    stereo=DatasetInfo(
        basepath='stereo',
        frame_height=128,
        frame_width=384,
        num_channels=3
    ),

    sequence=DatasetInfo(
        basepath='sequence',
        frame_height=128,
        frame_width=384,
        num_channels=3
    ),

    mixed=DatasetInfo(
        basepath='*',
        frame_height=128,
        frame_width=384,
        num_channels=3
    )
)
_MODES = ('train', 'valid', 'test')


def _get_dataset_files(dateset_info, mode, root):
    basepath = dateset_info.basepath
    base = os.path.join(root, basepath, mode, "*.tfrecord")
    return tf.data.Dataset.list_files(base)


def _convert_frame_data(data):
    decoded_frame = tf.image.decode_image(data)
    return tf.cast(decoded_frame, tf.float32) / 255


class DataReader(object):
    def __init__(self, dataset, batch_size, root, mode='train'):

        self._dataset_info = _DATASETS[dataset]

        with tf.device('/cpu'):
            file_names = _get_dataset_files(self._dataset_info, mode, root)
            dataset = tf.data.TFRecordDataset(
                file_names, num_parallel_reads=os.cpu_count())
            dataset = dataset.map(
                self._parse_function, num_parallel_calls=os.cpu_count())
            dataset = dataset.batch(batch_size, drop_remainder=True)
            self.data = dataset.repeat().make_one_shot_iterator()

    def read(self):
        return self.data.get_next()

    def _parse_function(self, raw_data):
        feature_map = {
            'frame0': tf.FixedLenFeature([], dtype=tf.string),
            'frame1': tf.FixedLenFeature([], dtype=tf.string)
        }
        example = tf.parse_single_example(raw_data, feature_map)
        frames = self._preprocess_frames(example)
        return frames

    def _preprocess_frames(self, example):
        frame0 = _convert_frame_data(example['frame0'])
        frame1 = _convert_frame_data(example['frame1'])

        frame0 = tf.reshape(
            frame0, [
                self._dataset_info.frame_height,
                self._dataset_info.frame_width,
                self._dataset_info.num_channels
            ]
        )
        frame1 = tf.reshape(
            frame1, [
                self._dataset_info.frame_height,
                self._dataset_info.frame_width,
                self._dataset_info.num_channels
            ]
        )
        return frame0, frame1
