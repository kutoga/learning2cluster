import numpy as np

from keras.layers import Reshape, Concatenate, Bidirectional, LSTM, Dense, BatchNormalization, \
    Activation, TimeDistributed
import keras.backend as K

from core.nn.helper import slice_layer
from impl.nn.base.simple_loss.simple_loss_cluster_nn import SimpleLossClusterNN

class ClusterNNTry01(SimpleLossClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, output_dense_units=1024,
                 cluster_count_dense_layers=1, output_dense_layers=1, cluster_count_dense_units=1024,
                 state_size=16, global_state_size=256, iterations=3, f_compare_dense_units=256):
        super().__init__(data_provider, input_count, embedding_nn)

        # Network parameters
        self.__state_size = state_size
        self.__global_state_size = global_state_size
        self.__iterations = iterations
        self.__f_compare_dense_units = f_compare_dense_units
        self.__output_dense_units = output_dense_units
        self.__cluster_count_dense_layers = cluster_count_dense_layers
        self.__cluster_count_dense_units = cluster_count_dense_units
        self.__output_dense_layers = output_dense_layers

    def _build_network(self, network_input, network_output, additional_network_outputs, debug_output):
        cluster_counts = list(self.data_provider.get_cluster_counts())

        # The simple loss cluster NN requires a specific output: a list of softmax distributions
        # First in this list are all softmax distributions for k=k_min for each object, then for k=k_min+1 for each
        # object etc. At the end, there is the cluster count output.

        # First we get an embedding for the network inputs
        embeddings = self._get_embedding(network_input)

        def merge_list_to_tensor(lst):
            lst_shape = lst[0].shape
            lst_size = int(str(np.prod(lst_shape[1:]))) # int(str(...))? We have to prevent the Dimension type (somehow...): We like int
            lst_reshaper = Reshape((1, lst_size))
            lst_reshaped = [lst_reshaper(item) for item in lst]
            return Concatenate(axis=1)(lst_reshaped)
            # return lst_reshaped

        # Create initial embedding states (we just use a zero state)
        embeddings_state_init = self._s_layer(
            'embeddings_states_init', lambda name:
            Dense(self.__state_size, bias_initializer='zeros', kernel_initializer='zeros', trainable=False, name=name)
        )
        embeddings_states = list(map(embeddings_state_init, embeddings))

        # Create an initial global state (we just use a zero state)
        global_state = self._s_layer(
            'global_state_init', lambda name:
            Dense(self.__global_state_size, bias_initializer='zeros', kernel_initializer='zeros', trainable=False, name=name)
        )(embeddings[0])

        # Predefine some "functions"
        f_compare = self._s_layer(
            'f_compare', lambda name:
            Dense(self.__f_compare_dense_units, name=name)
        )
        f_bdlstm_merge = self._s_layer(
            'f_bdlstm_merge', lambda name:
            Bidirectional(LSTM(self.__state_size), name=name)
        )
        f_bdlstm_merge_d = self._s_layer(
            'f_bdlstm_merge_d', lambda name:
            Dense(self.__state_size, name=name)
        )
        f_global_state_bdlstm_merge = self._s_layer(
            'f_global_state_bdlstm_merge', lambda name:
            Bidirectional(LSTM(self.__global_state_size), name=name)
        )
        f_global_state_update = self._s_layer(
            'f_global_state_update', lambda name:
            Dense(self.__global_state_size, name=name)
        )

        # Execute all iterations
        for i in range(self.__iterations):
            new_embeddings_states = []

            for e_i in range(self.input_count):

                processed_data = []
                for e_j in range(self.input_count):
                    if e_i == e_j:
                        continue

                    # Get all input layers
                    inputs = [
                        embeddings[e_i], embeddings_states[e_i],
                        embeddings[e_j], embeddings_states[e_j],
                        global_state
                    ]

                    # Merge these inputs
                    merged = Concatenate()([
                        Reshape((-1,))(inp) for inp in inputs
                    ])

                    # Execute f_compare
                    processed = f_compare(merged)
                    processed = BatchNormalization()(processed)
                    processed = Activation('relu')(processed)

                    # Collect the data
                    processed_data.append(processed)

                # Do an lstm over processed data
                processed_data = merge_list_to_tensor(processed_data)# merge_list_to_tensor(processed_data)
                processed_data = f_bdlstm_merge(processed_data)
                processed_data = BatchNormalization()(processed_data)
                processed_data = f_bdlstm_merge_d(processed_data)
                processed_data = BatchNormalization()(processed_data)
                processed_data = Activation('relu')(processed_data)

                new_embeddings_states.append(processed_data)
            embeddings_states = new_embeddings_states

            # Update the global state
            c_embeddings = merge_list_to_tensor(embeddings)
            c_states = merge_list_to_tensor(embeddings_states)
            c_in = Concatenate(axis=2)([c_embeddings, c_states])

            global_state = f_global_state_bdlstm_merge(c_in)
            global_state = BatchNormalization()(global_state)
            global_state = f_global_state_update(global_state)
            global_state = BatchNormalization()(global_state)
            global_state = Activation('relu')(global_state)


        # Get the current embeddings states
        embeddings_processed_reshaper = Reshape((1, self.__state_size))
        embeddings_processed = list(map(embeddings_processed_reshaper, embeddings_states))

        # Create now two outputs: The cluster count and for each cluster count / object combination a softmax distribution.
        # These outputs are independent of each other, therefore it doesn't matter which is calculated first. Let us start
        # with the cluster count / object combinations.

        # First prepare some generally required layers
        layers = []
        for i in range(self.__output_dense_layers):
            layers += [
                self._s_layer('output_dense{}'.format(i), lambda name: Dense(self.__output_dense_units, name=name)),
                self._s_layer('output_batch'.format(i), lambda name: BatchNormalization(name=name)),
                self._s_layer('output_relu'.format(i), lambda name: Activation('relu', name=name))
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
        # cluster_count = self._s_layer('cluster_count_LSTM_merge', lambda name: Bidirectional(LSTM(self.__lstm_units), name=name)(embeddings_merged))
        # cluster_count = self._s_layer('cluster_count_LSTM_merge_batch', lambda name: BatchNormalization(name=name))(cluster_count)
        cluster_count = global_state
        for i in range(self.__cluster_count_dense_layers):
            cluster_count = self._s_layer('cluster_count_dense{}'.format(i), lambda name: Dense(self.__cluster_count_dense_units, name=name))(cluster_count)
            cluster_count = self._s_layer('cluster_count_batch{}'.format(i), lambda name: BatchNormalization(name=name))(cluster_count)
            cluster_count = self._s_layer('cluster_count_relu{}'.format(i), lambda name: Activation('relu', name=name))(cluster_count)

        # The next layer is an output-layer, therefore the name must not be formatted
        cluster_count = self._s_layer(
            'cluster_count_output',
            lambda name: Dense(len(cluster_counts), activation='softmax', name=name),
            format_name=False
        )(cluster_count)
        additional_network_outputs['cluster_count_output'] = cluster_count

        network_output.append(cluster_count)

        return True