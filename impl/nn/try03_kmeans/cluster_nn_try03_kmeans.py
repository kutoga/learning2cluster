from itertools import chain

import numpy as np

from keras.layers import Reshape, Concatenate, Bidirectional, LSTM, Dense, BatchNormalization, \
    Activation, merge, Lambda
import keras.backend as K

from core.nn.helper import slice_layer, gaussian_random_layer
from impl.nn.base.simple_loss.simple_loss_cluster_nn import SimpleLossClusterNN


class ClusterNNTry03KMeans(SimpleLossClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, lstm_units=64, output_dense_units=512,
                 cluster_count_dense_layers=1, lstm_layers=5, output_dense_layers=1, cluster_count_dense_units=512,
                 weighted_classes=False, kmeans_itrs=3):
        super().__init__(data_provider, input_count, embedding_nn, weighted_classes)

        # Network parameters
        self.__lstm_layers = lstm_layers
        self.__lstm_units = lstm_units
        self.__kmeans_itrs = kmeans_itrs

        self.__output_dense_units = output_dense_units
        self.__cluster_count_dense_layers = cluster_count_dense_layers
        self.__cluster_count_dense_units = cluster_count_dense_units
        self.__output_dense_layers = output_dense_layers

    def _build_network(self, network_input, network_output, additional_network_outputs, debug_output):
        cluster_counts = list(self.data_provider.get_cluster_counts())

        # The simple loss cluster NN requires a specific output: a list of softmax distributions
        # First in this list are all softmax distributions for k=k_min for each object, then for k=k_min+1 for each
        # object etc. At the end, there is the cluster count output.

        def add_dbg_output(name, layer):
            debug_output.append(Activation('linear', name=name)(layer))

        # First we get an embedding for the network inputs
        embeddings = self._get_embedding(network_input)

        # Reshape all embeddings to 1d vectors
        # embedding_shape = self._embedding_nn.model.layers[-1].output_shape
        # embedding_size = np.prod(embedding_shape[1:])
        embedding_shape = embeddings[0].shape
        embedding_size = int(str(np.prod(embedding_shape[1:])))
        embedding_reshaper = self._s_layer('embedding_reshape', lambda name: Reshape((1, embedding_size), name=name))
        embeddings_reshaped = [embedding_reshaper(embedding) for embedding in embeddings]

        # Merge all embeddings to one tensor
        embeddings_merged = self._s_layer('embeddings_merge', lambda name: Concatenate(axis=1, name=name))(embeddings_reshaped)

        # Use now some BDLSTM-layer to process all embeddings.
        processed = embeddings_merged
        for i in range(self.__lstm_layers):
            processed = self._s_layer(
                'LSTM_proc_{}'.format(i), lambda name: Bidirectional(LSTM(self.__lstm_units, return_sequences=True), name=name)
            )(processed)
            processed = self._s_layer(
                'LSTM_proc_{}_batch'.format(i), lambda name: BatchNormalization(name=name)
            )(processed)

        # Batch normalize everything if not already done
        add_dbg_output('EMBEDDINGS', processed)
        if self.__lstm_layers == 0:
            processed = BatchNormalization()(processed)
        add_dbg_output('EMBEDDINGS_NORMALIZED', processed)

        # Split the tensor to seperate layers
        embeddings_processed = [self._s_layer('slice_{}'.format(i), lambda name: slice_layer(processed, i, name)) for i in range(len(network_input))]

        # Create now two outputs: The cluster count and for each cluster count / object combination a softmax distribution.
        # These outputs are independent of each other, therefore it doesn't matter which is calculated first. Let us start
        # with the cluster count / object combinations.

        # # First prepare some generally required layers
        # layers = []
        # for i in range(self.__output_dense_layers):
        #     layers += [
        #         self._s_layer('output_dense{}'.format(i), lambda name: Dense(self.__output_dense_units, name=name)),
        #         self._s_layer('output_batch'.format(i), lambda name: BatchNormalization(name=name)),
        #         self._s_layer('output_relu'.format(i), lambda name: Activation('relu', name=name))
        #     ]
        # cluster_softmax = {
        #     k: self._s_layer('softmax_cluster_{}'.format(k), lambda name: Dense(k, activation='softmax', name=name)) for k in cluster_counts
        # }
        #
        # # Create now the outputs
        # clusters_output = additional_network_outputs['clusters'] = {}
        # for i in range(len(embeddings_processed)):
        #     embedding_proc = embeddings_processed[i]
        #
        #     # Add the required layers
        #     for layer in layers:
        #         embedding_proc = layer(embedding_proc)
        #
        #     input_clusters_output = clusters_output['input{}'.format(i)] = {}
        #     for k in cluster_counts:
        #
        #         # Create now the required softmax distributions
        #         output_classifier = cluster_softmax[k](embedding_proc)
        #         input_clusters_output['cluster{}'.format(k)] = output_classifier
        #         network_output.append(output_classifier)

        # Define a distance-function
        d = lambda x, y: K.sqrt(K.sum(K.square(x - y))) # euclidean distance

        def euclideanDistance(inputs, squared=False):
            if (len(inputs) != 2):
                raise 'oops'

            # For better gradient flow: remove the sqrt
            output = K.sum(K.square(inputs[0] - inputs[1]), axis=-1)
            if not squared:
                output = K.sqrt(output)
            output = K.expand_dims(output, 1)
            return output

        # Apply k-means for each possible k
        assert self.__kmeans_itrs > 0
        cluster_vector_size = (self.__lstm_units * 2) if self.__lstm_layers > 0 else embedding_size
        cluster_assignements = {}
        for k in cluster_counts:

            # Create initial cluster centers
            # clusters = [self._s_layer(
            #     'k_{}_init_{}'.format(k, i), lambda name: gaussian_random_layer((1, cluster_vector_size), name=name, only_execute_for_training=False)
            # )(embeddings[0]) for i in range(k)]

            # # Use the first n input points
            clusters = embeddings_processed[:k]

            for i in range(len(clusters)):
                add_dbg_output('INIT_CLUSTER_{}'.format(i), clusters[i])

            # Cluster-assignements
            current_cluster_assignements = None

            # Idee zum neuen Mittelwert finden:
            # - Nicht mit Softmax rechnen sondern nur noch mit der Distanz
            # - Clusterzentrum mit Punkten Gewichten, falls die Distanz zum Cluster vom Punkt minimal ist (also kein anderer Cluster näher ist)

            # Do all iterations
            for i in range(self.__kmeans_itrs):

                def get_val_at(input, i_i):
                    def at(val, indices):
                        for index in indices:
                            val = val[index]
                        return val
                    return Lambda(lambda x: at(x, i_i), output_shape=(1,))(input)

                # Recalculate the cluster centers (if required)
                if i > 0:
                    clusters_old = clusters
                    clusters = []
                    for c_i in range(k):
                        c = 0
                        s = 0
                        for e_i in range(len(embeddings_processed)):
                            t = get_val_at(current_cluster_assignements[e_i], [1, c_i])
                            c = Lambda(lambda x: c + x * t, output_shape=(1, cluster_vector_size))(embeddings_processed[e_i])
                            # c = Lambda(lambda x: c + x * 10. * K.relu(t + 0.1 - K.max(current_cluster_assignements[e_i])), output_shape=(1, cluster_vector_size))(embeddings_processed[e_i])
                            s = Lambda(lambda x: x + s, output_shape=(1,))(t)

                            # t = current_cluster_assignements[e_i][c_i]
                            # c += t * embeddings_processed[e_i]
                            # s += t
                        c = Lambda(lambda x: x / s, output_shape=(1, cluster_vector_size))(c)

                        # c = c / s

                        c = Lambda(lambda x: 0 + x, output_shape=(1, cluster_vector_size))(c)
                        add_dbg_output("k{}_ITR{}_ci{}".format(k, i, c_i), c)
                        clusters.append(c)

                # Recalculate the assigned cluster centers for each embedding
                cluster_assignements[k] = []
                current_cluster_assignements = cluster_assignements[k]
                e_i = 0
                for e in embeddings_processed:

                    # Calculate all distances to all cluster centers
                    k_distances = [
                        Lambda(lambda x: euclideanDistance(x, True), output_shape=lambda x: (x[0][0], 1))([e, cluster])

                        # merge([e, cluster], mode=euclideanDistance, output_shape=lambda x: (x[0][0], 1))
                        for cluster in clusters
                        # self._s_layer('kmeans_d', lambda name: Merge([]))
                    ]

                    # Merge the distances and calculate a softmax
                    k_distances = Concatenate()(k_distances)
                    k_distances = Reshape((k,))(k_distances)
                    add_dbg_output("k{}_ITR{}_e_i{}_DISTANCES_PLAIN".format(k, i, e_i), k_distances)
                    # k_distances = Lambda(lambda x: 1 / (0.01 + x))(k_distances)
                    k_distances = Lambda(lambda x: -(1+x)**2)(k_distances)
                    k_distances = Activation('softmax')(k_distances)

                    # # Dirty tune the softmax a bit
                    # k_distances = Lambda(lambda x: (x / K.max(x))**4)(k_distances)

                    # Save the new distances
                    current_cluster_assignements.append(k_distances)
                    add_dbg_output("k{}_ITR{}_e_i{}_DISTANCES".format(k, i, e_i), k_distances)

                    e_i += 1

        # Reshape all softmax layers
        for k in cluster_counts:
            cluster_assignements[k] = [
                Reshape((1, k))(s) for s in cluster_assignements[k]
            ]

        # Append all softmax layers to the network output
        network_output += chain.from_iterable(map(
            lambda k: cluster_assignements[k],
            cluster_counts
        ))

        # Create the additional cluster output
        clusters_output = additional_network_outputs['clusters'] = {}
        for i in range(self.input_count):
            input_clusters_output = clusters_output['input{}'.format(i)] = {}
            for k in cluster_counts:
                input_clusters_output['cluster{}'.format(k)] = cluster_assignements[k][i]
                add_dbg_output('RESULT_INPUT{}_K{}'.format(i, k), cluster_assignements[k][i])

        # Calculate the real cluster count
        cluster_count = self._s_layer('cluster_count_LSTM_merge', lambda name: Bidirectional(LSTM(self.__lstm_units), name=name)(embeddings_merged))
        cluster_count = self._s_layer('cluster_count_LSTM_merge_batch', lambda name: BatchNormalization(name=name))(cluster_count)
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