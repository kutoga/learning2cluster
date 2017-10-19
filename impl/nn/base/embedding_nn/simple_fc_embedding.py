import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Layer, Flatten

from core.nn.embedding_nn import EmbeddingNN


class SimpleFCEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, hidden_layers=1, hidden_activation='relu', final_activation='sigmoid',
                 batch_norm_for_init_layer=True, batch_norm_for_final_layer=False, batch_norm_after_activation=False):
        """
        Important: Do NOT change the default values, they are required because of compatibility issues.

        :param output_size:
        :param hidden_layers:
        :param hidden_activation: a number of hidden layers or an array with the numbers of units for each hidden layer
        :param final_activation:
        :param batch_norm_for_final_layer:
        """
        super().__init__()

        self._output_size = output_size
        self._hidden_layers = hidden_layers
        self._hidden_activation = hidden_activation
        self._final_activation = final_activation
        self._batch_norm_for_init_layer = batch_norm_for_init_layer
        self._batch_norm_for_final_layer = batch_norm_for_final_layer
        self._batch_norm_after_activation = batch_norm_after_activation

    def _build_model(self, input_shape):
        input_points = np.product(input_shape)

        model = Sequential(name=self._get_name('Model'))
        if len(input_shape) > 1:
            model.add(self._s_layer('flatten_init', lambda name: Flatten(name=name, input_shape=input_shape)))
        else:
            # The first layer always requires the networks input shape. Create a dummy layer (its the easiest way)
            model.add(self._s_layer('dummy_init', lambda name: Activation('linear', name=name, input_shape=input_shape)))

        # Add BatchNorm if required
        if self._batch_norm_for_init_layer:
            model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name)))

        if isinstance(self._hidden_layers, list):
            dimensions = self._hidden_layers
        else:
            dimensions = [input_points * 32] * self._hidden_layers

        for i in range(len(dimensions)):
            model.add(self._s_layer('dense{}'.format(i), lambda name: Dense(dimensions[i], name=name, kernel_regularizer=self.regularizer)))
            batch_norm = self._s_layer('batch{}'.format(i), lambda name: BatchNormalization(name=name))
            if not self._batch_norm_after_activation:
                model.add(batch_norm)
            if isinstance(self._hidden_activation, Layer):
                model.add(self._hidden_activation)
            else:
                model.add(self._s_layer('activation{}'.format(i), lambda name: Activation(self._hidden_activation, name=name)))
            if self._batch_norm_after_activation:
                model.add(batch_norm)

        # TODO: change name (currently unchanged, because of compatibility issues; if the name is changed, old weights no longer can be loaded)
        model.add(self._s_layer('output', lambda name: Dense(self._output_size, name=name, kernel_regularizer=self.regularizer)))

        batch_norm = self._s_layer('output_batch', lambda name: BatchNormalization(name=name))
        if self._batch_norm_for_final_layer and not self._batch_norm_after_activation:
            model.add(batch_norm)
        model.add(self._s_layer('output_activation', lambda name: Activation(self._final_activation, name=name)))
        if self._batch_norm_for_final_layer and self._batch_norm_after_activation:
            model.add(batch_norm)

        return model
