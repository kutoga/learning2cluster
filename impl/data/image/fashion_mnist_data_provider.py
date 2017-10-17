import numpy as np
from impl.data.external import fashion_mnist

from impl.data.image.image_data_provider import ImageDataProvider

class FashionMNISTDataProvider(ImageDataProvider):
    def __init__(self, train_classes=[0, 2, 3, 4, 6, 7], validate_classes=[1, 5, 8, 9], test_classes=[1, 5, 8, 9],
                 min_cluster_count=None, max_cluster_count=None):
        super().__init__(train_classes, validate_classes, test_classes, min_cluster_count, max_cluster_count, center_data=True)

    def _get_img_data_shape(self):
        return (28, 28, 1)

    def _load_data(self):

        # Load all records
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Merge them (we split them by classes)
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        # Reshape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [0, 1]
        x = self._scale_data(x)

        # Split the records by classes and store it
        return {i: x[y == i] for i in np.unique(y)}

if __name__ == '__main__':
    dp = FashionMNISTDataProvider()

