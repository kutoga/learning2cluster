import numpy as np
from impl.data.misc.simple_class_based_dataset import load_data

from impl.data.image.image_data_provider import ImageDataProvider

class FaceScrubDataProvider(ImageDataProvider):
    """
    A DataProvider for the FaceScrub dataset. The dataset may be downloaded and up to some kind pre-processed with
    the following scripts:
    https://github.com/kutoga/facescrub

    Alternative:
    Dear Benjamin Meier,

    thank you for your interest in the FaceScrub dataset.
    You can download it from this link: http://vintage.winkler.site/faceScrub.zip
    The password for decrypting the zip file is: ICIP'14

    Best regards,

    Stefan Winkler
    ADSC
    """
    def __init__(self, dataset_dir, target_img_size=(48, 48), train_classes=None, validate_classes=None, test_classes=None,
                 min_cluster_count=None, max_cluster_count=None, min_element_count_per_cluster=1, additional_augmentor=None):
        self.__dataset_dir = dataset_dir
        self.__img_size = target_img_size
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count,
                         center_data=True, random_mirror_images=True, min_element_count_per_cluster=min_element_count_per_cluster,
                         additional_augmentor=additional_augmentor)

    def _get_img_data_shape(self):
        return self.__img_size + (3,)

    def _load_data(self):

        # Load all records
        (x_train, y_train) = load_data(self.__dataset_dir, 'facescrub', self.__img_size)

        # Merge them (we split them by classes)
        x = x_train
        y = y_train

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [0, 1]
        x = self._scale_data(x)

        # Split the records by classes and store it
        return {i: x[y == i] for i in np.unique(y)}

if __name__ == '__main__':
    dp = FaceScrubDataProvider('E:\\tmp\\test\\facescrub_128x128\\')

