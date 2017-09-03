import numpy as np

from keras.layers import Reshape, Concatenate, Bidirectional, LSTM, Dense, BatchNormalization, \
    Activation, TimeDistributed, RepeatVector, add

from core.nn.helper import slice_layer
from impl.nn.base.simple_loss.simple_loss_cluster_nn import SimpleLossClusterNN

class ClusterNNTry02Mod01(SimpleLossClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None,
                 f_update_global_state_lstm_units=512,
                 f_update_global_state_dense_layer_count=2, f_update_local_state_units=256,
                 f_update_local_state_dense_layer_count=2, f_cluster_count_units=512,
                 f_cluster_count_dense_layer_count=2, f_cluster_assignment_units=512,
                 f_cluster_assignment_dense_layer_count=2,
                 state_size=64, global_state_size=512, iterations=4):
        super().__init__(data_provider, input_count, embedding_nn)

        # Network parameters
        self.__local_state_size = state_size
        self.__global_state_size = global_state_size
        self.__iterations = iterations
        self.__f_update_global_state_lstm_units = f_update_global_state_lstm_units
        self.__f_update_global_state_dense_layer_count = f_update_global_state_dense_layer_count
        self.__f_update_local_state_units = f_update_local_state_units
        self.__f_update_local_state_dense_layer_count = f_update_local_state_dense_layer_count
        self.__f_cluster_count_units = f_cluster_count_units
        self.__f_cluster_count_dense_layer_count = f_cluster_count_dense_layer_count
        self.__f_cluster_assignment_units = f_cluster_assignment_units
        self.__f_cluster_assignment_dense_layer_count = f_cluster_assignment_dense_layer_count

    def _build_network(self, network_input, network_output, additional_network_outputs, debug_output):
        cluster_counts = list(self.data_provider.get_cluster_counts())

        # f_update_global_state
        def f_update_global_state(x, s, g):
            xs = self._s_layer(
                'f_update_global_state_CONCAT', lambda name:
                Concatenate(axis=2, name=name)
            )([x, s])
            res = self._s_layer(
                'f_update_global_state_BDLSTM', lambda name:
                Bidirectional(LSTM(self.__f_update_global_state_lstm_units), name=name)
            )(xs)
            res = self._s_layer(
                'f_update_global_state_BDLSTM_BNORM', lambda name:
                BatchNormalization(name=name)
            )(res)
            res = self._s_layer(
                'f_update_global_state_gCONCAT', lambda name:
                Concatenate(axis=1, name=name)
            )([g, res])
            for i in range(self.__f_update_global_state_dense_layer_count):
                res = self._s_layer(
                    'f_update_global_state_DENSE{}'.format(i), lambda name:
                    Dense(self.__global_state_size, name=name)
                )(res)
                res = self._s_layer(
                    'f_update_global_state_DENSE_BNORM{}'.format(i), lambda name:
                    BatchNormalization(name=name)
                )(res)
                res = self._s_layer(
                    'f_update_global_state_RELU{}'.format(i), lambda name:
                    Activation('relu' if i < self.__f_update_global_state_dense_layer_count - 1 else 'sigmoid')
                )(res)
            return res

        # f_update_local_state
        def f_update_local_state(x, s, g):
            g = self._s_layer(
                'f_update_local_state_REPEAT', lambda name:
                RepeatVector(self.input_count, name=name)
            )(g)
            xg = self._s_layer(
                'f_update_local_state_CONCAT', lambda name:
                Concatenate(axis=2, name=name)
            )([x, s, g])
            for i in range(self.__f_update_local_state_dense_layer_count):
                xg = self._s_layer(
                    'f_update_local_state_DENSE{}'.format(i), lambda name:
                    TimeDistributed(Dense(self.__f_update_local_state_units), name=name)
                )(xg)
                xg = self._s_layer(
                    'f_update_local_state_BNORM{}'.format(i), lambda name:
                    BatchNormalization(name=name)
                )(xg)
                xg = self._s_layer(
                    'f_update_local_state_RELU{}'.format(i), lambda name:
                    Activation('relu')
                )(xg)
            xg = self._s_layer(
                'f_update_local_state_OUT_DENSE', lambda name:
                TimeDistributed(Dense(self.__local_state_size), name=name)
            )(xg)
            xg = self._s_layer(
                'f_update_local_state_OUT_BNORM', lambda name:
                BatchNormalization(name=name)
            )(xg)
            xg = self._s_layer(
                'f_update_local_state_OUT_SIG', lambda name:
                Activation('sigmoid')
            )(xg)
            return xg

        # f_cluster_count
        def f_cluster_count(g):
            for i in range(self.__f_cluster_count_dense_layer_count):
                g = self._s_layer(
                    'f_cluster_count_DENSE{}'.format(i), lambda name:
                    Dense(self.__f_cluster_count_units, name=name)
                )(g)
                g = self._s_layer(
                    'f_cluster_count_BNORM{}'.format(i), lambda name:
                    BatchNormalization(name=name)
                )(g)
                g = self._s_layer(
                    'f_cluster_count_RELU{}'.format(i), lambda name:
                    Activation('relu')
                )(g)
            return g

        # f_cluster_assignment
        def f_cluster_assignment(x, g):
            g = self._s_layer(
                'f_cluster_assignment_gRESHAPE', lambda name:
                Reshape((1, self.__global_state_size), name=name)
            )(g)
            xg = self._s_layer(
                'f_cluster_assignment_CONCAT', lambda name:
                Concatenate(axis=2, name=name)
            )([x, g])
            for i in range(self.__f_cluster_assignment_dense_layer_count):
                xg = self._s_layer(
                    'f_cluster_assignment_DENSE{}'.format(i), lambda name:
                    Dense(self.__f_cluster_assignment_units, name=name)
                )(xg)
                xg = self._s_layer(
                    'f_cluster_assignment_BNORM{}'.format(i), lambda name:
                    BatchNormalization(name=name)
                )(xg)
                xg = self._s_layer(
                    'f_cluster_assignment_RELU{}'.format(i), lambda name:
                    Activation('relu')
                )(xg)
            return xg

        # First we get an embedding for the network inputs
        embeddings = self._get_embedding(network_input)

        # Reshape all embeddings to 1d vectors
        embedding_shape = embeddings[0].shape
        embedding_size = int(str(np.prod(embedding_shape[1:])))
        embedding_reshaper = self._s_layer('embedding_reshape', lambda name: Reshape((1, embedding_size), name=name))
        embeddings_reshaped = [embedding_reshaper(embedding) for embedding in embeddings]

        # Merge all embeddings to one tensor
        embeddings = self._s_layer('embeddings_merge', lambda name: Concatenate(axis=1, name=name))(embeddings_reshaped)

        # Create empty initial local states
        s = self._s_layer(
            'local_states_init', lambda name:
            TimeDistributed(Dense(self.__local_state_size, kernel_initializer='zeros', bias_initializer='zeros', trainable=False), name=name)
        )(embeddings)

        # Initialize the global state: Use a simple BDLSTM for this
        g = self._s_layer(
            'global_state_init_BDLSTM', lambda name:
            Bidirectional(LSTM(self.__f_update_global_state_lstm_units), name=name)
        )(embeddings)
        g = self._s_layer(
            'global_state_init_DENSE', lambda name:
            Dense(self.__global_state_size, name=name)
        )(g)
        g = self._s_layer(
            'global_state_init_BNORM', lambda name:
            BatchNormalization(name=name)
        )(g)
        g = self._s_layer(
            'global_state_init_RELU', lambda name:
            Activation('relu', name=name)
        )(g)

        # Do the iterations
        for i in range(self.__iterations):

            # Update the gloabl state
            g = add([g, f_update_global_state(embeddings, s, g)])

            # Update the local states
            s = add([s, f_update_local_state(embeddings, s, g)])

        # Get the processed embeddings (=states) as a list
        embeddings_processed = [self._s_layer('slice_{}'.format(i), lambda name: slice_layer(s, i, name)) for i in range(len(network_input))]

        # Create now two outputs: The cluster count and for each cluster count / object combination a softmax distribution.
        # These outputs are independent of each other, therefore it doesn't matter which is calculated first. Let us start
        # with the cluster count / object combinations.

        # First prepare some generally required layers
        cluster_softmax = {
            k: self._s_layer('softmax_cluster_{}'.format(k), lambda name: Dense(k, activation='softmax', name=name)) for k in cluster_counts
        }

        # Create now the outputs
        clusters_output = additional_network_outputs['clusters'] = {}
        for i in range(len(embeddings_processed)):
            embedding_proc = embeddings_processed[i]
            embedding_proc = f_cluster_assignment(embedding_proc, g)

            input_clusters_output = clusters_output['input{}'.format(i)] = {}
            for k in cluster_counts:

                # Create now the required softmax distributions
                output_classifier = cluster_softmax[k](embedding_proc)
                input_clusters_output['cluster{}'.format(k)] = output_classifier
                network_output.append(output_classifier)

        # Calculate the real cluster count
        # cluster_count = self._s_layer('cluster_count_LSTM_merge', lambda name: Bidirectional(LSTM(self.__lstm_units), name=name)(embeddings_merged))
        # cluster_count = self._s_layer('cluster_count_LSTM_merge_batch', lambda name: BatchNormalization(name=name))(cluster_count)
        cluster_count = f_cluster_count(g)

        # The next layer is an output-layer, therefore the name must not be formatted
        cluster_count = self._s_layer(
            'cluster_count_output',
            lambda name: Dense(len(cluster_counts), activation='softmax', name=name),
            format_name=False
        )(cluster_count)
        additional_network_outputs['cluster_count_output'] = cluster_count

        network_output.append(cluster_count)

        return True