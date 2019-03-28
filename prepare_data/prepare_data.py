import os
import glob
import shutil
import cv2
import argparse
import tensorflow as tf
from zipfile import ZipFile
from urllib.request import urlretrieve
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


url_template_raw = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{0}/{0}_sync.zip"
)
url_template_kitti = "https://s3.eu-central-1.amazonaws.com/avg-kitti/{}.zip"


kitti_raw_city = [
    "2011_09_26_drive_0001",
    "2011_09_26_drive_0002",
    "2011_09_26_drive_0005",
    "2011_09_26_drive_0009",
    "2011_09_26_drive_0011",
    "2011_09_26_drive_0013",
    "2011_09_26_drive_0014",
    "2011_09_26_drive_0017",
    "2011_09_26_drive_0018",
    "2011_09_26_drive_0048",
    "2011_09_26_drive_0051",
    "2011_09_26_drive_0056",
    "2011_09_26_drive_0057",
    "2011_09_26_drive_0059",
    "2011_09_26_drive_0060",
    "2011_09_26_drive_0084",
    "2011_09_26_drive_0091",
    "2011_09_26_drive_0093",
    "2011_09_26_drive_0095",
    "2011_09_26_drive_0096",
    "2011_09_26_drive_0104",
    "2011_09_26_drive_0106",
    "2011_09_26_drive_0113",
    "2011_09_26_drive_0117",
    "2011_09_28_drive_0001",
    "2011_09_28_drive_0002",
    "2011_09_29_drive_0026",
    "2011_09_29_drive_0071"
]

kitti_raw_residential = [
    "2011_09_26_drive_0019",
    "2011_09_26_drive_0020",
    "2011_09_26_drive_0022",
    "2011_09_26_drive_0023",
    "2011_09_26_drive_0035",
    "2011_09_26_drive_0036",
    "2011_09_26_drive_0039",
    "2011_09_26_drive_0046",
    "2011_09_26_drive_0061",
    "2011_09_26_drive_0064",
    "2011_09_26_drive_0079",
    "2011_09_26_drive_0086",
    "2011_09_26_drive_0087",
    "2011_09_30_drive_0018",
    "2011_09_30_drive_0020",
    "2011_09_30_drive_0027",
    "2011_09_30_drive_0028",
    "2011_09_30_drive_0033",
    "2011_09_30_drive_0034",
    "2011_10_03_drive_0027",
    "2011_10_03_drive_0034"
]

kitti_raw_road = [
    "2011_09_26_drive_0015",
    "2011_09_26_drive_0027",
    "2011_09_26_drive_0028",
    "2011_09_26_drive_0029",
    "2011_09_26_drive_0032",
    "2011_09_26_drive_0052",
    "2011_09_26_drive_0070",
    "2011_09_26_drive_0101",
    "2011_09_29_drive_0004",
    "2011_09_30_drive_0016",
    "2011_10_03_drive_0042",
    "2011_10_03_drive_0047"
]


kitti_datasets = [
    "data_scene_flow",
    "data_depth_annotated"
]


def maybe_download(root, files, url_template):
    if not os.path.exists(root):
        os.makedirs(root)

    for f in tqdm(files):
        path = os.path.join(root, f + ".zip")
        url = url_template.format(f)
        if not os.path.exists(path):
            urlretrieve(url, path)


def extract_raw(root, files):
    save_path = os.path.join(root, "raw_sequences")
    tmp_path = os.path.join(root, "tmp")
    os.makedirs(save_path)

    for f in tqdm(files):
        path = os.path.join(root, f + ".zip")
        z = ZipFile(path, 'r')
        z.extractall(tmp_path)

        sequence_path = os.path.join(tmp_path, f[:10], f + "_sync")
        target_path = os.path.join(save_path, f)
        shutil.move(sequence_path, target_path)
        shutil.rmtree(tmp_path)


def extract_scene_flow(root, file):
    path = os.path.join(root, file + ".zip")
    z = ZipFile(path, 'r')
    z.extractall()


def load_image(addr):
    im = cv2.imread(addr)
    im = cv2.resize(im, (384, 128))
    im_str = cv2.imencode('.jpg', im)[1].tostring()
    return im_str


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def make_record(x, n, path):
    i, (frame0_path, frame1_path) = x

    frame0 = load_image(frame0_path)
    frame1 = load_image(frame1_path)

    feature = {
        'frame0': _bytes_feature([frame0]),
        'frame1': _bytes_feature([frame1]),
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))

    filename = "{}-of-{}.tfrecord".format(i + 1, n)
    filepath = os.path.join(path, filename)
    with tf.python_io.TFRecordWriter(filepath) as writer:
        writer.write(example.SerializeToString())


def make_training_set_stereo(root):
    save_path = os.path.join(root, "stereo", "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sequences = os.path.join(root, "raw_sequences")
    frame0_paths = sorted(
        glob.glob(os.path.join(sequences, "*/image_02/data/*.png"))
    )
    frame1_paths = sorted(
        glob.glob(os.path.join(sequences, "*/image_03/data/*.png"))
    )
    assert len(frame0_paths) == len(frame1_paths)

    p = Pool()
    n = len(frame0_paths)
    q = enumerate(zip(frame0_paths, frame1_paths))
    f = partial(make_record, n=n, path=save_path)

    for _ in tqdm(p.imap_unordered(f, q), smoothing=0, total=n):
        pass


def make_training_set_sequence(root):
    save_path = os.path.join(root, "sequence", "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frame0_paths = []
    frame1_paths = []

    for dir in glob.glob(os.path.join(root, "raw_sequences/*")):
        paths = sorted(glob.glob(os.path.join(dir, "image_02/data/*.png")))
        frame0_paths += paths[::2][:-1]
        frame1_paths += paths[::2][1:]
        paths = sorted(glob.glob(os.path.join(dir, "image_03/data/*.png")))
        frame0_paths += paths[::2][:-1]
        frame1_paths += paths[::2][1:]

    assert len(frame0_paths) == len(frame1_paths)

    p = Pool()
    n = len(frame0_paths)
    q = enumerate(zip(frame0_paths, frame1_paths))
    f = partial(make_record, n=n, path=save_path)

    for _ in tqdm(p.imap_unordered(f, q), smoothing=0, total=n):
        pass


def get_args():
    parser = argparse.ArgumentParser(
        description="Download and Preprocess KITTI data")
    parser.add_argument(
        'save_path',
        help='Path to store downloaded data. Should have 400GB of free space.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    root = args.save_path

    print("Maybe downloading kitti data; this may take a while.")
    maybe_download(
        root,
        kitti_raw_city + kitti_raw_residential + kitti_raw_road,
        url_template_raw
    )
    maybe_download(
        root,
        kitti_datasets,
        url_template_kitti
    )

    print("Extracting data")
    extract_raw(
       root,
       kitti_raw_city + kitti_raw_residential + kitti_raw_road
    )

    print("Creating TFRecords")
    make_training_set_stereo(root)
    make_training_set_sequence(root)
