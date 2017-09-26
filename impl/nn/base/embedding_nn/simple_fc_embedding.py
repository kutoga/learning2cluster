import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation

from core.nn.embedding_nn import EmbeddingNN


class SimpleFCEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, hidden_layers=1, final_activation='sigmoid', batch_norm_for_final_layer=False):
        super().__init__()

        self._output_size = output_size
        self._hidden_layers = hidden_layers
        self._final_activation = final_activation
        self._batch_norm_for_final_layer = batch_norm_for_final_layer

    def _build_model(self, input_shape):
        input_points = np.product(input_shape)

        model = Sequential(name=self._get_name('Model'))
        model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name, input_shape=input_shape)))

        if isinstance(self._hidden_layers, list):
            dimensions = self._hidden_layers
        else:
            dimensions = [input_points * 32] * self._hidden_layers

        for i in range(len(dimensions)):
            model.add(self._s_layer('dense{}'.format(i), lambda name: Dense(dimensions[i], name=name)))
            model.add(self._s_layer('batch{}'.format(i), lambda name: BatchNormalization(name=name)))
            model.add(self._s_layer('relu{}'.format(i), lambda name: Activation('relu', name=name)))

        # TODO: change name (currently unchanged, because of compatibility issues; if the name is changed old weights no longer can be loaded)
        model.add(self._s_layer('output', lambda name: Dense(self._output_size, name=name)))

        if self._batch_norm_for_final_layer:
            model.add(self._s_layer('output_batch', lambda name: BatchNormalization(name=name)))
        model.add(self._s_layer('output_activation', lambda name: Activation(self._final_activation, name=name)))

        return model
