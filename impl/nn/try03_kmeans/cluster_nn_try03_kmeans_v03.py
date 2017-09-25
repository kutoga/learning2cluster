from itertools import chain

import numpy as np

from keras.layers import Reshape, Concatenate, Bidirectional, LSTM, Dense, BatchNormalization, \
    Activation, Lambda, add, Dropout, Flatten
import keras.backend as K
import tensorflow as tf

from core.nn.helper import slice_layer, gaussian_random_layer, concat_layer
from impl.nn.base.simple_loss.simple_loss_cluster_nn import SimpleLossClusterNN


class ClusterNNTry03KMeansV03(SimpleLossClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, lstm_units=64, output_dense_units=512,
                 cluster_count_dense_layers=1, lstm_layers=5, output_dense_layers=1, cluster_count_dense_units=512,
                 weighted_classes=False, kmeans_itrs=3, kmeans_input_dimension=2):
        super().__init__(data_provider, input_count, embedding_nn, weighted_classes)

        # Network parameters
        self.__lstm_layers = lstm_layers
        self.__lstm_units = lstm_units
        self.__kmeans_itrs = kmeans_itrs
        self.__kmeans_input_dimension = kmeans_input_dimension

        self.__output_dense_units = output_dense_units
        self.__cluster_count_dense_layers = cluster_count_dense_layers
        self.__cluster_count_dense_units = cluster_count_dense_units
        self.__output_dense_layers = output_dense_layers

    def _build_network(self, network_input, network_output, additional_network_outputs, debug_output, additional_prediction_outputs):
        cluster_counts = list(self.data_provider.get_cluster_counts())

        # The simple loss cluster NN requires a specific output: a list of softmax distributions
        # First in this list are all softmax distributions for k=k_min for each object, then for k=k_min+1 for each
        # object etc. At the end, there is the cluster count output.

        def point_preprocessor(p):
            return Lambda(lambda x: K.tanh(x))(p)

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

        # Prepare the embeddings for the kmeans input
        if self.__kmeans_input_dimension is not None:

            # We just update the input dimensions for the kmeans, therefore no activation function is used
            layers = [
                self._s_layer('kmeans_dimension_changer_dense', lambda name: Dense(self.__kmeans_input_dimension, name=name)),
                self._s_layer('kmeans_dimension_changer_batch', lambda name: BatchNormalization(name=name))
            ]
            for layer in layers:
                embeddings_processed = [layer(e) for e in embeddings_processed]

        # Apply the preprocessing for all embeddings (we force the network to have all values in the range [-1, 1]);
        # this makes kmeans easier
        embeddings_processed = [point_preprocessor(e) for e in embeddings_processed]
        add_dbg_output('EMBEDDINGS_PREPROCESSED', processed)
        additional_prediction_outputs.append({
            'name': 'embeddings processed',
            'layer': embeddings_processed
        })

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

        def get_trainable_weight_f(name):
            def get_name(layer_name):
                return "{}_{}".format(name, layer_name)
            def distance_f(input):
                dense_units = [64, 128, 256]
                nw = input
                for di in range(len(dense_units)):
                    nw = self._s_layer(get_name('dense{}'.format(di)), lambda name: Dense(dense_units[di], name=name))(nw)
                    nw = self._s_layer(get_name('batch{}'.format(di)), lambda name: BatchNormalization(name=name))(nw)
                    nw = self._s_layer(get_name('relu{}'.format(di)), lambda name: Activation('relu', name=name))(nw)
                weight = self._s_layer(get_name('output'), lambda name: Dense(1, name=name))(nw)
                return weight
            return distance_f

        # Apply k-means for each possible k
        assert self.__kmeans_itrs > 0
        cluster_vector_size = self.__kmeans_input_dimension if self.__kmeans_input_dimension is not None else embedding_size #(self.__lstm_units * 2) if self.__lstm_layers > 0 else embedding_size
        cluster_assignements = {}
        for k in cluster_counts:

            # Create initial cluster centers
            clusters = [self._s_layer(
                'k_{}_init_{}'.format(k, i), lambda name: gaussian_random_layer((1, cluster_vector_size), name=name, only_execute_for_training=False)
            )(embeddings[0]) for i in range(k)]

            # Apply the preprocessing
            clusters = [point_preprocessor(c) for c in clusters]


            # # # Use the first n input points
            # clusters = embeddings_processed[:k]


            # # Create initial cluster centers:
            # # 1) The first cluster center is the mean of all points
            # def l_mean(inputs):
            #     inputs_len = len(inputs)
            #     l = add(inputs)
            #     l = Lambda(lambda x: x / inputs_len)(l)
            #     return l
            # clusters = [l_mean(embeddings_processed)]
            # # 2) Create now all other cluster centers
            # if k > 1:
            #     for i in range(1, k):
            #         # Sum over all points
            #         c_s = 0
            #         s_s = 0
            #
            #         for e in embeddings_processed:
            #             # Get the distances to all points from the current embedding
            #             dists = []
            #             for c in clusters:
            #                 dists.append(Lambda(lambda x: euclideanDistance(x, False), output_shape=lambda x: (x[0][0], 1))([c, e]))
            #
            #             if len(dists) == 1:
            #                 dists = dists[0]
            #             else:
            #                 dists = Concatenate()(dists)
            #
            #             # Calculate a weight for this data point
            #             w = Lambda(lambda x: K.exp(10 * K.min(x, axis=0)))(dists)
            #
            #             # Update c_s and s_s
            #             c_s = Lambda(lambda x: c_s + w * x)(e)
            #             s_s = Lambda(lambda x: s_s + w)(e)
            #
            #         # Calculate the new cluster center
            #         c_s = Lambda(lambda x: c_s / s_s)(c_s)
            #
            #         # Append it to the clusters list
            #         clusters.append(c_s)

            # # Choose k random points and use them as initial clusters
            # c_i_embeddings = Concatenate(axis=1)(embeddings_processed)
            # c_i_embeddings = Lambda(lambda x: tf.random_shuffle(x))(c_i_embeddings) # non-differentiable;
            # clusters = [slice_layer(c_i_embeddings, i) for i in range(k)]

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
                    clusters = []
                    for c_i in range(k):
                        c = 0
                        s = 1e-8 # Avoid zero-divison
                        for e_i in range(len(embeddings_processed)):

                            # Reorder the t values: First the current target and then all other cluster centers
                            t_values = [Reshape((1, 1))(get_val_at(current_cluster_assignements[e_i], [1, c_i]))]
                            for c_j in range(k):
                                if c_i != c_j:
                                    t_values.append(Reshape((1, 1))(get_val_at(current_cluster_assignements[e_i], [1, c_j])))
                            t_values = Concatenate()(t_values)
                            t_values = Flatten()(t_values)

                            # Calculate now a weight
                            t = get_trainable_weight_f('cluster_to_point')(t_values)

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
                    k_distance_f = get_trainable_weight_f('embedding_move')

                    # Which inputs should be used? Currently the euclidean distance is used, but maybe also the cluster centers are
                    # good and valuable pices of information?
                    k_distances = list(map(k_distance_f, k_distances))

                    k_distances = Concatenate()(k_distances)
                    k_distances = Reshape((k,))(k_distances)
                    # add_dbg_output("k{}_ITR{}_e_i{}_DISTANCES_PLAIN".format(k, i, e_i), k_distances)
                    # k_distances = Lambda(lambda x: 1 / (0.01 + x))(k_distances)

                    # k_distances = Lambda(lambda x: -(1 + 3 * K.sqrt(x)) ** 3)(k_distances)

                    # cluster_assignements[p_i][c_i] = -min(500, (1 + 3 * np.sqrt(d)) ** 3)  # + 3/(1.+d)

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