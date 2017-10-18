import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Layer, Bidirectional, LSTM

from core.nn.embedding_nn import EmbeddingNN


class BDLSTMEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, bdlstm_layers_units=[64, 64], fc_layers_units=[10, 10],
                 hidden_activation='relu', final_activation='sigmoid',
                 batch_norm_for_init_layer=False, batch_norm_for_final_layer=False, batch_norm_after_activation=True,
                 batch_norm_after_bdlstm=False
                 ):
        super().__init__()

        self._output_size = output_size
        self._bdlstm_layers_units = bdlstm_layers_units
        self._fc_layers_units = fc_layers_units
        self._hidden_activation = hidden_activation
        self._final_activation = final_activation
        self._batch_norm_for_init_layer = batch_norm_for_init_layer
        self._batch_norm_for_final_layer = batch_norm_for_final_layer
        self._batch_norm_after_activation = batch_norm_after_activation
        self._batch_norm_after_bdlstm = batch_norm_after_bdlstm

    def _build_model(self, input_shape):

        model = Sequential(name=self._get_name('Model'))
        if self._batch_norm_for_init_layer:
            model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name, input_shape=input_shape)))
        else:
            # The first layer always requires the networks input shape. Create a dummy layer (its the easiest way)
            model.add(self._s_layer('dummy_init', lambda name: Activation('linear', name=name, input_shape=input_shape)))

        # It is just assumed that the input is 1D with N features; No flattening is done
        # (this is done to prevent undesired / unexpected behaviour)

        # Add all BDLSTM layers
        bdlstm_layer_count = len(self._bdlstm_layers_units)
        assert bdlstm_layer_count > 0
        for i in range(bdlstm_layer_count):
            units = self._bdlstm_layers_units[i]
            return_sequences = i < (bdlstm_layer_count - 1)

            model.add(self._s_layer(
                'bdlstm{}'.format(i),
                lambda name: Bidirectional(LSTM(units, return_sequences=return_sequences), name=name)
            ))
            if self._batch_norm_after_bdlstm:
                model.add(self._s_layer('batch{}'.format(i), lambda name: BatchNormalization(name=name)))

        # Add all fully connected layers
        for i in range(len(self._fc_layers_units)):
            model.add(self._s_layer('dense{}'.format(i), lambda name: Dense(self._fc_layers_units[i], name=name, kernel_regularizer=self.regularizer)))
            batch_norm = self._s_layer('batch{}'.format(i), lambda name: BatchNormalization(name=name))
            if not self._batch_norm_after_activation:
                model.add(batch_norm)
            if isinstance(self._hidden_activation, Layer):
                model.add(self._hidden_activation)
            else:
                model.add(self._s_layer('activation{}'.format(i), lambda name: Activation(self._hidden_activation, name=name)))
            if self._batch_norm_after_activation:
                model.add(batch_norm)

        model.add(self._s_layer('output_dense', lambda name: Dense(self._output_size, name=name, kernel_regularizer=self.regularizer)))

        batch_norm = self._s_layer('output_batch', lambda name: BatchNormalization(name=name))
        if self._batch_norm_for_final_layer and not self._batch_norm_after_activation:
            model.add(batch_norm)
        model.add(self._s_layer('output_activation', lambda name: Activation(self._final_activation, name=name)))
        if self._batch_norm_for_final_layer and self._batch_norm_after_activation:
            model.add(batch_norm)

        return model
