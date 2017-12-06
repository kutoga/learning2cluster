import gzip
import os
import tarfile
import itertools
import glob
import pickle
import gzip

import shutil

from scipy.io import loadmat
from scipy.misc import imread, imresize

from keras.utils.data_utils import get_file
import numpy as np

def load_data(top_dir, dataset_name=None, target_img_size=(48, 48), extensions=['.jpg', '.png']):
    """Loads a simple class based dataset from a directory structure like:
    top_dir/class_name/img.(jpg|png)

    If a dataset-name is provided, the loaded images will be cached.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train)`.
    """
    if dataset_name is not None:
        dirname = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'simple_ds', dataset_name))

        cached_file = os.path.join(dirname, 'img_{}x{}.cache.pkl.gz'.format(*target_img_size))
        if os.path.exists(cached_file):

            # The cache-file exists. Read it and return its content.
            with gzip.open(cached_file, "rb") as fh:
                (x_train, y_train) = pickle.load(fh)
                return (x_train, y_train)

    # Loading from the cache did not work: Load the dataset (and cache it if a name was defined)

    # Find all class names (=subdirectories)
    def get_immediate_subdirectories(a_dir):
        # See: https://stackoverflow.com/a/800201/916672
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]
    classes = get_immediate_subdirectories(top_dir)

    # Load all objects
    dataset = {}
    tot_files_loaded = 0
    for class_name in sorted(classes):
        objects = []

        # Find all files
        img_files = sorted(itertools.chain(*[
            glob.glob(os.path.join(top_dir, class_name, '*{}'.format(extension))) for extension in extensions
        ]))

        # Load and resize the files
        for img_file in img_files:
            print("Read and pre-process image file (already loaded {} files): {}".format(tot_files_loaded, img_file))
            img = imread(img_file, mode='RGB')
            img = imresize(img, target_img_size + (3,))
            objects.append(img)
            tot_files_loaded += 1

        if len(objects) > 0:
            dataset[class_name] = objects

    # Create now the resulting arrays
    records = sum(map(lambda l: len(l), dataset.values()))
    x_train = np.zeros((records,) + target_img_size + (3,), dtype=np.float32)
    y_train = np.zeros((records,), dtype=np.float32)
    class_names = sorted(dataset.keys())
    i = 0
    for c_i in range(len(class_names)):
        objects = dataset[class_names[c_i]]
        for object in objects:
            x_train[i] = object
            y_train[i] = c_i
            i += 1

    # Cache these records (if possible)
    if dataset_name != None:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with gzip.open(cached_file, "wb", compresslevel=6) as fh:
            pickle.dump((x_train, y_train), fh)

    return (x_train, y_train)


if __name__ == '__main__':
    load_data()

