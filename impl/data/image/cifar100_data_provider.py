import numpy as np
from keras.datasets import cifar100

from random import Random

from impl.data.image.image_data_provider import ImageDataProvider

class Cifar100DataProvider(ImageDataProvider):
    def __init__(self, train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None):
        if train_classes is None and validate_classes is None and test_classes is None:
            rand = Random()
            rand.seed(1337)
            classes = list(range(100))
            rand.shuffle(classes)
            train_classes_count = 80
            train_classes = classes[:train_classes_count]
            validate_classes = classes[train_classes_count:]
            test_classes = classes[train_classes_count:]
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count)
        self.center_data = True

    def _get_img_data_shape(self):
        return (32, 32, 3)

    def _load_data(self):

        # Load all records
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Merge them (we split them by classes)
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [-1, 1]
        x = self._scale_data(x)
        # x = x.astype(np.float32) / 255

        # Split the records by classes and store it
        y = y.reshape((y.shape[0],))
        return {i: x[y == i] for i in np.unique(y)}
