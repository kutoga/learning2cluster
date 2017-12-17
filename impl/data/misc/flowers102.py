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
    """Loads the Flowers102 dataset
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_valid, y_valid), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'flowers102')
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    base = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'
    files = ['102flowers.tgz', 'imagelabels.mat', 'setid.mat']

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

        print("Extract all flower images...")
        tarfile.open(paths[0]).extractall(tmp_dir)

        print("Load index definitions...")
        idx = loadmat(paths[2])
        idx_train = idx['trnid'][0] - 1
        idx_test = idx['tstid'][0] - 1
        idx_valid = idx['valid'][0] - 1

        print("Load labels...")
        img_labels = loadmat(paths[1])['labels'][0] - 1

        print("Load and pre-process image files...")
        img_files = sorted(glob.glob(os.path.join(tmp_dir, 'jpg', '*.jpg')))
        x = np.zeros((len(img_files),) + target_img_size + (3,), dtype=np.uint8)
        y = np.zeros((len(img_files)))
        for i in range(len(img_files)):
            img_file = img_files[i]
            print("Read and pre-process image file: {}".format(img_file))
            img = imread(img_file)
            img = imresize(img, target_img_size + (3,))
            x[i] = img
            y[i] = img_labels[i]

        print("Split the data...")
        x_train = x[idx_train]
        y_train = y[idx_train]
        x_valid = x[idx_valid]
        y_valid = y[idx_valid]
        x_test = x[idx_test]
        y_test = y[idx_test]

        print("Delete temporary files")
        shutil.rmtree(tmp_dir)

        print("Save the data")
        data = ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))
        with open(cached_file, "wb") as fh:
            pickle.dump(data, fh, protocol=4)

    with open(cached_file, "rb") as fh:
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pickle.load(fh)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


if __name__ == '__main__':
    load_data()

