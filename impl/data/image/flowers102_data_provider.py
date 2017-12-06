import numpy as np
from random import Random

from impl.data.misc import flowers102

from impl.data.image.image_data_provider import ImageDataProvider


class Flowers102DataProvider(ImageDataProvider):
    def __init__(self, train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None, target_img_size=(48, 48),
                 min_element_count_per_cluster=1, additional_augmentor=None):
        self._target_img_size = target_img_size
        if train_classes is None and validate_classes is None and test_classes is None:
            rand = Random()
            rand.seed(1337)
            classes = list(range(102))
            rand.shuffle(classes)
            train_classes_count = int(0.8 * len(classes))
            train_classes = classes[:train_classes_count]
            validate_classes = classes[train_classes_count:]
            test_classes = classes[train_classes_count:]
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count,
                         center_data=True, random_mirror_images=True, min_element_count_per_cluster=min_element_count_per_cluster,
                         additional_augmentor=additional_augmentor)

    def _get_img_data_shape(self):
        return self._target_img_size + (3,)

    def _load_data(self):

        # Load all records
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = flowers102.load_data(self._target_img_size)

        # Merge them (we split them by classes)
        x = np.concatenate((x_train, x_valid, x_test))
        y = np.concatenate((y_train, y_valid, y_test))

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [0, 1]
        x = self._scale_data(x)

        # Split the records by classes and store it
        return {i: x[y == i] for i in np.unique(y)}

if __name__ == '__main__':
    dp = Flowers102DataProvider()

