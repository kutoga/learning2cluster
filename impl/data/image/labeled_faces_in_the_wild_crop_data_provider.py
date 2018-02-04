import numpy as np
from impl.data.misc.simple_class_based_dataset import load_data

from impl.data.image.image_data_provider import ImageDataProvider

class LabeledFacesInTheWildCropDataProvider(ImageDataProvider):
    def __init__(self, dataset_dir, target_img_size=(48, 48), train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None, min_element_count_per_cluster=1, additional_augmentor=None,
                 use_all_classes_for_train_test_validation=False, min_images_per_class=1, allow_resampling=True):
        self.__img_size = target_img_size
        self.__min_images_per_class = min_images_per_class
        self.__dataset_dir = dataset_dir
        self.__img_size = target_img_size
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count,
                         center_data=True, random_mirror_images=True, min_element_count_per_cluster=min_element_count_per_cluster,
                         additional_augmentor=additional_augmentor, use_all_classes_for_train_test_validation=use_all_classes_for_train_test_validation,
                         allow_resampling=allow_resampling)

    def _get_img_data_shape(self):
        return self.__img_size + (3,)

    def _load_data(self):

        # Load all records
        (x_train, y_train) = load_data(self.__dataset_dir, 'lfw_crop', self.__img_size)

        # Merge them (we split them by classes)
        x = x_train
        y = y_train

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [0, 1]
        x = self._scale_data(x)

        # Split the records by classes and store it
        data = {i: x[y == i] for i in np.unique(y)}

        # Remove classes with to less records
        if self.__min_images_per_class is not None and self.__min_images_per_class > 1:
            new_data = {}
            counter = 0
            for i in data.keys():
                if len(data[i]) >= self.__min_images_per_class:
                    new_data[counter] = data[i]
                    counter += 1
            data = new_data

        return data