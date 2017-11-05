import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Layer, Bidirectional, LSTM, Flatten, TimeDistributed, \
    Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Dropout

from core.nn.embedding_nn import EmbeddingNN


class CnnBDLSTMEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, bdlstm_layers_units=[64, 64], fc_layers_units=[10, 10],
                 cnn_layers_per_block=2, block_feature_counts=[16, 32, 64],
                 hidden_activation='relu', final_activation='sigmoid', cnn_filter_size=3, max_pooling_size=2, max_pooling_stride=2,
                 dimensionality='2d', dropout_after_max_pooling=[],
                 batch_norm_for_init_layer=False, batch_norm_for_final_layer=False, batch_norm_after_activation=True,
                 batch_norm_after_bdlstm=False
                 ):
        super().__init__()

        self._output_size = output_size
        self._bdlstm_layers_units = bdlstm_layers_units
        self._cnn_layers_per_block = cnn_layers_per_block
        self._block_feature_counts = block_feature_counts
        self._fc_layers_units = fc_layers_units
        self._hidden_activation = hidden_activation
        self._final_activation = final_activation
        self._batch_norm_for_init_layer = batch_norm_for_init_layer
        self._batch_norm_for_final_layer = batch_norm_for_final_layer
        self._batch_norm_after_activation = batch_norm_after_activation
        self._batch_norm_after_bdlstm = batch_norm_after_bdlstm
        self._cnn_filter_size = cnn_filter_size
        self._max_pooling_size = max_pooling_size
        self._max_pooling_stride = max_pooling_stride
        self._dimensionality = dimensionality
        self._dropout_after_max_pooling = dropout_after_max_pooling

    def _build_model(self, input_shape):
        dimensionality = self._dimensionality
        if dimensionality == 'auto':
            if len(input_shape) == 2:
                dimensionality = '1d'
            elif len(input_shape) == 3:
                dimensionality = '2d'
            else:
                raise ValueError("Cannot detect dimensionality from input shape.")

        # We requires a time and a feature dimension (if you only have (n,)-shaped data, you have to reshape it)
        assert len(input_shape) >= 2

        model = Sequential(name=self._get_name('Model'))

        # Create a function to add a dropout layer
        def add_dropout_if_required(name, rate):
            if rate is None or rate <= 0.:
                return
            model.add(self._s_layer(name, lambda name: Dropout(rate, name=name)))

        if self._batch_norm_for_init_layer:
            model.add(self._s_layer('batch_init', lambda name: BatchNormalization(name=name, input_shape=input_shape)))
        else:
            # The first layer always requires the networks input shape. Create a dummy layer (its the easiest way)
            model.add(self._s_layer('dummy_init', lambda name: Activation('linear', name=name, input_shape=input_shape)))

        # Add all convolutional layers
        for i in range(len(self._block_feature_counts)):
            block_feature_count = self._block_feature_counts[i]
            for j in range(self._cnn_layers_per_block):

                # Add a convolutional layer
                if dimensionality == '1d':
                    model.add(self._s_layer('cnn1d{}_{}'.format(i, j), lambda name: Convolution1D(block_feature_count, self._cnn_filter_size, padding='same', name=name, kernel_regularizer=self.regularizer)))
                elif dimensionality == '2d':
                    model.add(self._s_layer('cnn2d{}_{}'.format(i, j), lambda name: Convolution2D(block_feature_count, (self._cnn_filter_size, self._cnn_filter_size), padding='same', name=name, kernel_regularizer=self.regularizer)))
                else:
                    raise ValueError("Invalid dimensionality: {}".format(dimensionality))

                batch_norm = self._s_layer('cnn{}_{}_batch'.format(i, j), lambda name: BatchNormalization(name=name))
                if not self._batch_norm_after_activation:
                    model.add(batch_norm)

                # Add the activation
                if isinstance(self._hidden_activation, Layer):
                    model.add(self._hidden_activation)
                else:
                    model.add(self._s_layer('cnn{}_{}_activation'.format(i, j), lambda name: Activation(self._hidden_activation, name=name)))

                if self._batch_norm_after_activation:
                    model.add(batch_norm)

            # Add max pooling
            if dimensionality == '1d':
                model.add(self._s_layer('max1{}'.format(i), lambda name: MaxPooling1D(name=name, pool_size=self._max_pooling_size, strides=self._max_pooling_stride)))
            elif dimensionality == '2d':
                model.add(self._s_layer('max2{}'.format(i), lambda name: MaxPooling2D(name=name, pool_size=(self._max_pooling_size, self._max_pooling_size), strides=(self._max_pooling_stride, self._max_pooling_stride))))
            else:
                raise ValueError("Invalid dimensionality: {}".format(dimensionality))

            # Add dropout if required
            add_dropout_if_required('dropout_max_{}'.format(i), self._dropout_after_max_pooling[i])

        # If the input dimensionality id 2d, we have to flatten the input
        if dimensionality == '2d':
            model.add(self._s_layer(
                'flatten_init',
                lambda name: TimeDistributed(Flatten(), name=name)
            ))

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
