import numpy as np
from random import Random

from impl.data.misc.simple_class_based_dataset import load_data

from impl.data.image.image_data_provider import ImageDataProvider

class TinyImageNetDataProvider(ImageDataProvider):
    def __init__(self, dataset_dir, target_img_size=(48, 48), train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None, min_element_count_per_cluster=1, additional_augmentor=None,
                 use_all_classes_for_train_test_validation=False):
        self.__dataset_dir = dataset_dir
        self.__img_size = target_img_size
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count,
                         center_data=True, random_mirror_images=True, min_element_count_per_cluster=min_element_count_per_cluster,
                         additional_augmentor=additional_augmentor, use_all_classes_for_train_test_validation=use_all_classes_for_train_test_validation)

    def _get_img_data_shape(self):
        return self.__img_size + (3,)

    def _load_data(self):

        # Load all records
        (x, y) = load_data(self.__dataset_dir, 'tiny_image_net', self.__img_size)

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [-1, 1]
        x = self._scale_data(x)
        # x = x.astype(np.float32) / 255

        # Split the records by classes and store it
        y = y.reshape((y.shape[0],))
        return {i: x[y == i] for i in np.unique(y)}
