import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Convolution1D, Convolution2D, MaxPooling1D,\
    MaxPooling2D, Flatten, Layer, Dropout

from core.nn.embedding_nn import EmbeddingNN


class CnnEmbedding(EmbeddingNN):
    def __init__(self, output_size=4, cnn_layers_per_block=2, block_feature_counts=[16, 32, 64],
                 fc_layer_feature_counts=[512, 512],
                 hidden_activation='relu', final_activation='tanh',
                 batch_norm_for_init_layer=False, batch_norm_for_final_layer=False, dimensionality='2d',
                 batch_norm_after_activation=True, cnn_filter_size=3, max_pooling_size=2, max_pooling_stride=2,
                 dropout_init=None, dropout_after_max_pooling=[], dropout_after_fc=[]):
        """

        :param output_size:
        :param cnn_layers_per_block:
        :param block_feature_counts:
        :param fc_layer_feature_counts:
        :param hidden_activation:
        :param final_activation:
        :param batch_norm_for_init_layer:
        :param batch_norm_for_final_layer:
        :param dimensionality:
        :param batch_norm_after_activation:
        :param cnn_filter_size:
        :param max_pooling_size:
        :param max_pooling_stride:
        :param dropout_init: Dropout directly after the initial layer.
        :param dropout_after_max_pooling: Dropout rate after max pooling layers. This value might be a single scaler from 0.0 to 1.0, None or a list with the dropout for the nth max pooling layer.
        :param dropout_after_fc: Dropout rate after fully connected layers. This value might be a single scaler from 0.0 to 1.0, None or a list with the dropout for the nth fully connected layer.
        """
        super().__init__()
        self._output_size = output_size
        self._cnn_layers_per_block = cnn_layers_per_block
        self._block_feature_counts = block_feature_counts
        self._fc_layer_feature_counts = fc_layer_feature_counts
        self._hidden_activation = hidden_activation
        self._final_activation = final_activation
        self._batch_norm_for_init_layer = batch_norm_for_init_layer
        self._batch_norm_for_final_layer = batch_norm_for_final_layer
        self._dimensionality = dimensionality
        self._batch_norm_after_activation = batch_norm_after_activation
        self._cnn_filter_size = cnn_filter_size
        self._max_pooling_size = max_pooling_size
        self._max_pooling_stride = max_pooling_stride

        # Define the dropouts
        def get_dropout_lst(input_def, target_list_len, default_value=None):
            if isinstance(input_def, list):
                res = input_def[:target_list_len]
                res = res + [default_value] * (target_list_len - len(res))
            else:
                res = [input_def] * target_list_len
            return res
        self._dropout_init = dropout_init
        self._dropout_after_max_pooling = get_dropout_lst(dropout_after_max_pooling, len(self._block_feature_counts))
        self._dropout_after_fc = get_dropout_lst(dropout_after_fc, len(self._fc_layer_feature_counts))

        # Fill the dropout

        assert self._dimensionality in ['1d', '2d', 'auto']

    def _build_model(self, input_shape):
        dimensionality = self._dimensionality
        if dimensionality == 'auto':
            if len(input_shape) == 2:
                dimensionality = '1d'
            elif len(input_shape) == 3:
                dimensionality = '2d'
            else:
                raise ValueError("Cannot detect dimensionality from input shape.")

        # Initialize the (sequential) model
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

        # Add an initial dropout layer
        add_dropout_if_required('dropout_init', self._dropout_init)

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

        # Flatten
        model.add(self._s_layer('flatten', lambda name: Flatten(name=name)))

        # Add all fully connected layers
        for i in range(len(self._fc_layer_feature_counts)):
            fc_layer_feature_count = self._fc_layer_feature_counts[i]
            model.add(self._s_layer('fcl{}'.format(i), lambda name: Dense(fc_layer_feature_count, name=name, kernel_regularizer=self.regularizer)))
            batch_norm = self._s_layer('fcl{}_batch'.format(i), lambda name: BatchNormalization(name=name))
            if not self._batch_norm_after_activation:
                model.add(batch_norm)
            model.add(self._s_layer('fcl{}_activation'.format(i), lambda name: Activation(self._hidden_activation, name=name)))
            if self._batch_norm_after_activation:
                model.add(batch_norm)

            # Add dropout if required
            add_dropout_if_required('dropout_fc_{}'.format(i), self._dropout_after_fc[i])

        # Add the output
        model.add(self._s_layer('output_dense', lambda name: Dense(self._output_size, name=name, kernel_regularizer=self.regularizer)))
        batch_norm = self._s_layer('output_batch', lambda name: BatchNormalization(name=name))
        if self._batch_norm_for_final_layer and not self._batch_norm_after_activation:
            model.add(batch_norm)
        if isinstance(self._final_activation, Layer):
            model.add(self._final_activation)
        else:
            model.add(self._s_layer('output_activation'.format(i, j), lambda name: Activation(self._final_activation, name=name)))
        if self._batch_norm_for_final_layer and self._batch_norm_after_activation:
            model.add(batch_norm)

        return model
