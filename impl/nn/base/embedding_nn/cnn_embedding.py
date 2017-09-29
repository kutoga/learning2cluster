import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Convolution2D, MaxPooling2D

from core.nn.embedding_nn import EmbeddingNN


class CnnEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, cnn_layer_per_block=2, block_feature_counts=[16, 32, 64],
                 fc_layer_feature_counts=[512, 512],
                 hidden_activation='relu', final_activation='sigmoid',
                 batch_norm_for_init_layer=False, batch_norm_for_final_layer=False):
        super().__init__()
        self._output_size = output_size
        self._cnn_layer_per_block = cnn_layer_per_block
        self._block_feature_counts = block_feature_counts
        self._fc_layer_feature_counts = fc_layer_feature_counts
        self._hidden_activation = hidden_activation
        self._final_activation = final_activation
        self._batch_norm_for_init_layer = batch_norm_for_init_layer
        self._batch_norm_for_final_layer = batch_norm_for_final_layer

    def _build_model(self, input_shape):

        # Initialize the (sequential) model
        model = Sequential(name=self._get_name('Model'))
        if self._batch_norm_for_init_layer:
            model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name, input_shape=input_shape)))
        else:
            # The first layer always requires the networks input shape. Create a dummy layer (its the easiest way)
            model.add(self._s_layer('dummy_init', lambda name: Activation('linear', name=name, input_shape=input_shape)))

        # Add all convolutional layers
        for i in range(len(self._block_feature_counts)):
            block_feature_count = self._block_feature_counts[i]
            for j in range(self._cnn_layer_per_block):
                model.add(self._s_layer('cnn{}_{}'.format(i, j), lambda name: Convolution2D(block_feature_count, (3, 3), padding='same', name=name)))
                model.add(self._s_layer('cnn{}_{}_batch'.format(i, j), lambda name: BatchNormalization(name=name)))
                model.add(self._s_layer('cnn{}_{}_activation'.format(i, j), lambda name: Activation(self._hidden_activation, name=name)))
            model.add(self._s_layer('max{}'.format(i), lambda name: MaxPooling2D(name=name)))

        # Add all fully connected layers
        for i in range(len(self._fc_layer_feature_counts)):
            fc_layer_feature_count = self._fc_layer_feature_counts[i]
            model.add(self._s_layer('fcl{}'.format(i), lambda name: Dense(fc_layer_feature_count, name=name)))
            model.add(self._s_layer('fcl{}_activation'.format(i), lambda name: Activation(self._hidden_activation, name=name)))
            model.add(self._s_layer('fcl{}_batch'.format(i), lambda name: BatchNormalization(name=name)))

        # Add the output
        model.add(self._s_layer('output_dense', lambda name: Dense(self._output_size, name=name)))
        if self._batch_norm_for_final_layer:
            model.add(self._s_layer('output_batch', lambda name: BatchNormalization(name=name)))
        model.add(self._s_layer('output_activation', lambda name: Activation(self._final_activation, name=name)))

        return model
