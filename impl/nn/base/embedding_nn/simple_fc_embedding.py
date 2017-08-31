import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation

from core.nn.embedding_nn import EmbeddingNN


class SimpleFCEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, hidden_layers=1):
        EmbeddingNN.__init__(self)

        self._output_size = output_size
        self._hidden_layers = hidden_layers

    def _build_model(self, input_shape):
        input_points = np.product(input_shape)

        model = Sequential(name=self._get_name('Model'))
        model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name, input_shape=input_shape)))

        for i in range(self._hidden_layers):
            model.add(self._s_layer('dense{}'.format(i), lambda name: Dense(input_points * 32, name=name)))
            model.add(self._s_layer('batch{}'.format(i), lambda name: BatchNormalization(name=name)))
            model.add(self._s_layer('relu{}'.format(i), lambda name: Activation('relu', name=name)))

        model.add(self._s_layer('output', lambda name: Dense(self._output_size, activation='sigmoid', name=name)))

        return model