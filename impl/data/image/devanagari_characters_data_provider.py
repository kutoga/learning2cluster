import numpy as np
import pickle
from os import path

from random import Random

from impl.data.image.image_data_provider import ImageDataProvider

#
# The data.csv file can be downloaded from here:
# https://www.kaggle.com/rishianand/devanagari-character-set/data
#

class DevangariCharactersDataProvider(ImageDataProvider):
    def __init__(self, data_csv_path, train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None):
        self._data_csv_path = data_csv_path
        if train_classes is None and validate_classes is None and test_classes is None:
            rand = Random()
            rand.seed(1337)
            classes = self._load_class_names()
            rand.shuffle(classes)
            train_classes_count = int(0.8 * len(classes))
            train_classes = classes[:train_classes_count]
            validate_classes = classes[train_classes_count:]
            test_classes = classes[train_classes_count:]
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count, center_data=True)

    def _get_img_data_shape(self):
        return (32, 32, 1)

    def _load_class_names(self):
        class_names = set()
        pkl_cache_file = self._data_csv_path + ".class.pkl"

        if path.exists(pkl_cache_file):
            with open(pkl_cache_file, "rb") as fh:
                class_names = pickle.load(fh)
        else:
            with open(self._data_csv_path, "r") as fh:
                next(fh) # skip the header line
                for line in fh:
                    record = line.split(',')
                    assert len(record) == 1025
                    class_names.add(record[-1])
            class_names = list(sorted(class_names))
            with open(pkl_cache_file, "wb") as fh:
                pickle.dump(class_names, fh)
        return class_names

    def _load_data(self):
        data = {}
        pkl_cache_file = self._data_csv_path + ".pkl"

        if path.exists(pkl_cache_file):
            with open(pkl_cache_file, "rb") as fh:
                data = pickle.load(fh)
        else:
            with open(self._data_csv_path, "r") as fh:
                next(fh) # skip the header line
                for line in fh:
                    record = line.split(',')
                    assert len(record) == 1025
                    img = np.zeros((1, 32, 32, 1), dtype=np.float32)
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            img[0, y, x, 0] = float(record[img.shape[0] * y + x])
                    cls = record[-1]
                    if cls not in data:
                        data[cls] = []
                    data[cls].append(img)
            for cls in list(data.keys()):
                data[cls] = self._scale_data(np.concatenate(data[cls]))
            with open(pkl_cache_file, "wb") as fh:
                pickle.dump(data, fh)

        return data

if __name__ == '__main__':
    dp = DevangariCharactersDataProvider("E:\\tmp\\test\\devanagari-character-set.csv")