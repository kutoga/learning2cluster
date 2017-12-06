from keras.layers import Reshape, Concatenate, Bidirectional, Dense, BatchNormalization, \
    Activation, Lambda, Add, LSTM
from keras.layers.advanced_activations import LeakyReLU

import numpy as np

from core.nn.helper import slice_layer, meier_cluster_cohesion, concat
from impl.nn.base.simple_loss.simple_loss_cluster_nn_v02 import SimpleLossClusterNN_V02

# LSTM = CuDNNLSTM

class ClusterNNCohesionTest(SimpleLossClusterNN_V02):
    def __init__(self, data_provider, input_count, embedding_nn=None, output_dense_units=512,
                 cluster_count_dense_layers=1, lstm_layers=5, output_dense_layers=1, cluster_count_dense_units=512,
                 weighted_classes=False, cluster_count_lstm_layers=2, cluster_count_lstm_units=64, internal_embedding_size=96,
                 kl_embedding_size=128, kl_divergence_factor=1.):
        super().__init__(data_provider, input_count, embedding_nn, weighted_classes)

        self.__internal_embedding_size = internal_embedding_size

        # Network parameters
        self.__lstm_layers = lstm_layers
        self.__output_dense_units = output_dense_units
        self.__cluster_count_lstm_layers = cluster_count_lstm_layers
        self.__cluster_count_lstm_units = cluster_count_lstm_units
        self.__cluster_count_dense_layers = cluster_count_dense_layers
        self.__cluster_count_dense_units = cluster_count_dense_units
        self.__output_dense_layers = output_dense_layers
        self.__kl_embedding_size = kl_embedding_size
        self.__kl_divergence_factor = kl_divergence_factor

    def _build_network(self, network_input, network_output, additional_network_outputs):
        cluster_counts = list(self.data_provider.get_cluster_counts())

        # The simple loss cluster NN requires a specific output: a list of softmax distributions
        # First in this list are all softmax distributions for k=k_min for each object, then for k=k_min+1 for each
        # object etc. At the end, there is the cluster count output.

        # First we get an embedding for the network inputs
        embeddings = self._get_embedding(network_input)

        # Reshape all embeddings to 1d vectors
        # embedding_shape = self._embedding_nn.model.layers[-1].output_shape
        # embedding_size = np.prod(embedding_shape[1:])
        embedding_shape = embeddings[0].shape
        embedding_size = int(str(np.prod(embedding_shape[1:])))
        embedding_reshaper = self._s_layer('embedding_reshape', lambda name: Reshape((1, embedding_size), name=name))
        embeddings_reshaped = [embedding_reshaper(embedding) for embedding in embeddings]

        cohesion_d_layer = self._s_layer('cohesion_d_layer', lambda name: Dense(2, activation='tanh', name=name))
        cohesion_embeddings = [cohesion_d_layer(embedding) for embedding in embeddings_reshaped]
        self._add_additional_prediction_output(
            concat(cohesion_embeddings, axis=1),
            'cohesion_embeddings'
        )
        self._register_additional_embedding_comparison_regularisation(
            'cluster_cohesion',

            # If the values are no longer used (as this is the case here), we need a dissimilarity weight, because
            # otherwise there is the trivial solution to map all inputs to 0
            lambda x, y, similar: meier_cluster_cohesion(x, y, similar, dissimilar_distance_weight=-1),

            cohesion_embeddings
        )

        # We need now the internal representation of the embeddings. This means we have to resize them.
        internal_embedding_size = self.__internal_embedding_size // 2 * 2
        embedding_internal_resizer = self._s_layer('internal_embedding_resize', lambda name: Dense(internal_embedding_size, name=name))
        embeddings_reshaped = [embedding_internal_resizer(embedding) for embedding in embeddings_reshaped]
        embedding_internal_resizer_act = LeakyReLU()
        embeddings_reshaped = [embedding_internal_resizer_act(embedding) for embedding in embeddings_reshaped]

        # Merge all embeddings to one tensor
        embeddings_merged = self._s_layer('embeddings_merge', lambda name: Concatenate(axis=1, name=name))(embeddings_reshaped)

        # Use now some lstm-layers
        processed = embeddings_merged
        for i in range(self.__lstm_layers):
            tmp = self._s_layer(
                'LSTM_proc_{}'.format(i), lambda name: Bidirectional(LSTM(internal_embedding_size // 2, return_sequences=True), name=name)
            )(processed)
            processed = Add()([processed, tmp])

        # Split the tensor to seperate layers
        embeddings_processed = [self._s_layer('slice_{}'.format(i), lambda name: slice_layer(processed, i, name)) for i in range(len(network_input))]

        # Create now two outputs: The cluster count and for each cluster count / object combination a softmax distribution.
        # These outputs are independent of each other, therefore it doesn't matter which is calculated first. Let us start
        # with the cluster count / object combinations.

        # First prepare some generally required layers
        layers = []
        for i in range(self.__output_dense_layers):
            layers += [
                self._s_layer('output_dense{}'.format(i), lambda name: Dense(self.__output_dense_units, name=name)),
                self._s_layer('output_batch'.format(i), lambda name: BatchNormalization(name=name)),
                LeakyReLU()
                # self._s_layer('output_relu'.format(i), lambda name: Activation(LeakyReLU(), name=name))
            ]
        cluster_softmax = {
            k: self._s_layer('softmax_cluster_{}'.format(k), lambda name: Dense(k, activation='softmax', name=name)) for k in cluster_counts
        }

        # Create now the outputs
        clusters_output = additional_network_outputs['clusters'] = {}
        for i in range(len(embeddings_processed)):
            embedding_proc = embeddings_processed[i]

            # Add the required layers
            for layer in layers:
                embedding_proc = layer(embedding_proc)

            input_clusters_output = clusters_output['input{}'.format(i)] = {}
            for k in cluster_counts:

                # Create now the required softmax distributions
                output_classifier = cluster_softmax[k](embedding_proc)
                input_clusters_output['cluster{}'.format(k)] = output_classifier
                network_output.append(output_classifier)

        # Calculate the real cluster count
        assert self.__cluster_count_lstm_layers >= 1
        cluster_count = embeddings_merged
        for i in range(self.__cluster_count_lstm_layers - 1):
            cluster_count = self._s_layer('cluster_count_LSTM{}'.format(i), lambda name: Bidirectional(LSTM(self.__cluster_count_lstm_units, return_sequences=True), name=name)(cluster_count))
            cluster_count = self._s_layer('cluster_count_LSTM{}_batch'.format(i), lambda name: BatchNormalization(name=name))(cluster_count)
        cluster_count = self._s_layer('cluster_count_LSTM_merge', lambda name: Bidirectional(LSTM(self.__cluster_count_lstm_units), name=name)(cluster_count))
        cluster_count = self._s_layer('cluster_count_LSTM_merge_batch', lambda name: BatchNormalization(name=name))(cluster_count)
        for i in range(self.__cluster_count_dense_layers):
            cluster_count = self._s_layer('cluster_count_dense{}'.format(i), lambda name: Dense(self.__cluster_count_dense_units, name=name))(cluster_count)
            cluster_count = self._s_layer('cluster_count_batch{}'.format(i), lambda name: BatchNormalization(name=name))(cluster_count)
            cluster_count = LeakyReLU()(cluster_count)
            # cluster_count = self._s_layer('cluster_count_relu{}'.format(i), lambda name: Activation(LeakyReLU(), name=name))(cluster_count)

        # The next layer is an output-layer, therefore the name must not be formatted
        cluster_count = self._s_layer(
            'cluster_count_output',
            lambda name: Dense(len(cluster_counts), activation='softmax', name=name),
            format_name=False
        )(cluster_count)
        additional_network_outputs['cluster_count_output'] = cluster_count

        network_output.append(cluster_count)

        return True