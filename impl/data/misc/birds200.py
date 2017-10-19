import gzip
import os
import tarfile
import glob
import pickle

import shutil

from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave

from keras.utils.data_utils import get_file
import numpy as np

def load_data(target_img_size=(48, 48)):
    """Loads the Flowers102 dataset
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_valid, y_valid), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'birds200')
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    # http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    base = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
    files = ['CUB_200_2011.tgz']

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
        tmp_data_dir = os.path.join(tmp_dir, 'CUB_200_2011')

        print("Extract all bird images...")
        tarfile.open(paths[0]).extractall(tmp_dir)

        def read_definition_file(file):
            with open(file, "r") as fh:
                return [line.strip().split(' ') for line in fh]

        # A hash that contains all images and all properties
        imgs = {}
        def get_img_dict(img_id):
            if not img_id in imgs:
                imgs[img_id] = {}
            return imgs[img_id]

        print("Load the list of images...")
        for img_id, path in read_definition_file(os.path.join(tmp_data_dir, 'images.txt')):
            get_img_dict(img_id)['path'] = path

        print("Load the classes...")
        for img_id, class_id in read_definition_file(os.path.join(tmp_data_dir, 'image_class_labels.txt')):
            get_img_dict(img_id)['class_id'] = int(class_id) - 1

        print("Load the bounding boxes...")
        for img_id, x, y, width, height in read_definition_file(os.path.join(tmp_data_dir, 'bounding_boxes.txt')):
            img_dict = get_img_dict(img_id)
            img_dict['bb_x'] = int(float(x))
            img_dict['bb_y'] = int(float(y))
            img_dict['bb_width'] = int(float(width))
            img_dict['bb_height'] = int(float(height))

        print("Load the train / test split...")
        for img_id, is_training_image in read_definition_file(os.path.join(tmp_data_dir, 'train_test_split.txt')):
            get_img_dict(img_id)['is_training_image'] = is_training_image

        print("Load and pre-process image files...")
        x = np.zeros((len(imgs),) + target_img_size + (3,), dtype=np.uint8)
        y = np.zeros((len(imgs)))
        idx_train = []
        idx_test = []
        img_ids = sorted(imgs.keys())
        for i in range(len(img_ids)):
            img_dict = imgs[img_ids[i]]
            print("Read and pre-process image file ({}/{}): {}".format(i + 1, len(img_ids), img_dict['path']))
            img = imread(os.path.join(tmp_data_dir, 'images', img_dict['path']), mode='RGB')
            img = img[
                img_dict['bb_y']:(img_dict['bb_y'] + img_dict['bb_height']),
                img_dict['bb_x']:(img_dict['bb_x'] + img_dict['bb_width']),
            ]
            img = imresize(img, target_img_size + (3,))
            # imsave(os.path.join(tmp_data_dir, 'images', img_dict['path'] + '_resized.png'), img)
            x[i] = img
            y[i] = img_dict['class_id']
            if img_dict['is_training_image']:
                idx_train.append(i)
            else:
                idx_test.append(i)

        print("Split the data...")
        x_train = x[idx_train]
        y_train = y[idx_train]
        x_test = x[idx_test]
        y_test = y[idx_test]

        print("Delete temporary files")
        shutil.rmtree(tmp_dir)

        print("Save the data")
        data = ((x_train, y_train), (x_test, y_test))
        with open(cached_file, "wb") as fh:
            pickle.dump(data, fh)

    with open(cached_file, "rb") as fh:
        (x_train, y_train), (x_test, y_test) = pickle.load(fh)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    load_data()

