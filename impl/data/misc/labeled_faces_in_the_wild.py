import gzip
import os
import tarfile
import glob
import pickle

import shutil

from scipy.io import loadmat
from scipy.misc import imread, imresize

from keras.utils.data_utils import get_file
import numpy as np

def load_data(target_img_size=(48, 48)):
    """Loads the Labeled Faces in the Wild (LFW) dataset
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train)`.
    """
    dirname = os.path.join('datasets', 'lfw')
    # http://vis-www.cs.umass.edu/lfw/lfw.tgz
    base = 'http://vis-www.cs.umass.edu/lfw/'
    files = ['lfw.tgz']

    # Download all files
    paths = []
    for file in files:
        paths.append(get_file(file, origin=base + file, cache_subdir=dirname))

    # Load all files
    assert len(target_img_size) == 2
    cached_file = os.path.expanduser(os.path.join('~', '.keras', dirname, 'img_{}x{}.cache.pkl'.format(*target_img_size)))

    # If required: Pre-process all files
    if not os.path.exists(cached_file):
        # The following code is partially based on:
        # https://github.com/Arsey/keras-transfer-learning-for-oxford102/blob/master/bootstrap.py

        tmp_dir = os.path.expanduser(os.path.join('~', '.keras', dirname, 'tmp'))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        print("Extract all face images...")
        tarfile.open(paths[0]).extractall(tmp_dir)

        data_dir = os.path.join(tmp_dir, 'lfw')
        classes = os.listdir(data_dir)
        data = {cls:[] for cls in classes}

        # Load all objects
        print("Load all images...")
        tot_records = 0
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            img_files = sorted(glob.glob(os.path.join(cls_dir, '*.jpg')))
            for img_file in img_files:
                print("Read and pre-process image file: {}".format(img_file))
                img = imread(img_file)
                img = imresize(img, target_img_size + (3,))
                data[cls].append(img)
                tot_records += 1

        # Create the resulting numpy arrays
        x_train = np.zeros((tot_records,) + target_img_size + (3,), dtype=np.uint8)
        y_train = np.zeros((tot_records,))

        # Go in a deterministic way through all classes (sort them first)
        classes = sorted(classes)
        d_i = 0 # data index
        for i in range(len(classes)):
            cls_data = data[classes[i]]
            for img in cls_data:
                x_train[d_i] = img
                y_train[d_i] = i
                d_i += 1

        print("Delete temporary files")
        shutil.rmtree(tmp_dir)

        print("Save the data")
        data = ((x_train, y_train))
        with open(cached_file, "wb") as fh:
            pickle.dump(data, fh, protocol=4)

    with open(cached_file, "rb") as fh:
        (x_train, y_train) = pickle.load(fh)

    return (x_train, y_train)


if __name__ == '__main__':
    load_data((128, 128))

