from inspect import signature
from time import time

import numpy as np

from keras.layers import Dot, Reshape, Activation, Lambda
from keras.losses import binary_crossentropy
import keras.backend as K

from core.nn.cluster_nn import ClusterNN
from core.nn.helper import filter_None, concat, concat_layer, create_weighted_binary_crossentropy, mean_confidence_interval


class SimpleLossClusterNN_V02(ClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, weighted_classes=False, cluster_n_output_loss='categorical_crossentropy',
                 use_cluster_count_loss=True, use_similarities_loss=True, include_input_count_in_name=True, similarities_loss_f=None):
        super().__init__(data_provider, input_count, embedding_nn, include_input_count_in_name=include_input_count_in_name)

        # For the loss all n outputs are compared with each other. More exactly:
        # n(n-1)/2 comparisons are required. If the following setting is True, then
        # each element is also comapred with itself. But why? This forces the network
        # to produce distributions with one large value.
        self._include_self_comparison = True

        self._weighted_classes = weighted_classes
        self._class_weights_approximator = 'simple_approximation' # Possible values: ['stochastic', 'simple_approximation]
        self._class_weights_post_processing_f = None # None == lambda x: x == identity function
        self._normalize_class_weights = True

        self._cluster_n_output_loss = cluster_n_output_loss

        self._additional_grouping_similarity_losses = []
        self._additional_regularisations = []
        self._additional_embedding_comparison_regularisations = []

        self._use_cluster_count_loss = use_cluster_count_loss
        self._use_similarities_loss = use_similarities_loss

        # If a custom similarities loss function shoudl be used, then it can be defined via similarities_loss_f.
        # Important: If such a loss function is defined then the class weights are ignored.
        self._similarities_loss_f = similarities_loss_f

    @property
    def use_cluster_count_loss(self):
        return self._use_cluster_count_loss

    @use_cluster_count_loss.setter
    def use_cluster_count_loss(self, use_cluster_count_loss):
        self._use_cluster_count_loss = use_cluster_count_loss

    @property
    def use_similarities_loss(self):
        return self._use_similarities_loss

    @use_similarities_loss.setter
    def use_similarities_loss(self, use_similarities_loss):
        self._use_similarities_loss = use_similarities_loss

    @property
    def include_self_comparison(self):
        return self._include_self_comparison

    @include_self_comparison.setter
    def include_self_comparison(self, include_self_comparison):
        self._include_self_comparison = include_self_comparison

    @property
    def weighted_classes(self):
        return self._weighted_classes

    @weighted_classes.setter
    def weighted_classes(self, weighted_classes):
        self._weighted_classes = weighted_classes

    @property
    def class_weights_approximation(self):
        return self._class_weights_approximator

    @class_weights_approximation.setter
    def class_weights_approximation(self, class_weights_approximator):
        self._class_weights_approximator = class_weights_approximator

    @property
    def class_weights_post_processing_f(self):
        return self._class_weights_post_processing_f

    @class_weights_post_processing_f.setter
    def class_weights_post_processing_f(self, class_weights_post_processing_f):
        self._class_weights_post_processing_f = class_weights_post_processing_f

    @property
    def normalize_class_weights(self):
        return self._normalize_class_weights

    @normalize_class_weights.setter
    def normalize_class_weights(self, normalize_class_weights):
        self._normalize_class_weights = normalize_class_weights

    @property
    def cluster_n_output_loss(self):
        return self._cluster_n_output_loss

    @cluster_n_output_loss.setter
    def cluster_n_output_loss(self, cluster_n_output_loss):
        self._cluster_n_output_loss = cluster_n_output_loss

    @property
    def similarities_loss_f(self):
        return self._similarities_loss_f

    @similarities_loss_f.setter
    def similarities_loss_f(self, similarities_loss_f):
        self._similarities_loss_f = similarities_loss_f

    def _register_additional_grouping_similarity_loss(self, name, loss_f, calculate_org_similarity_loss=True):
        """
        This losses give the possibility to add custom losses based on the similarity_loss (that still has to be defined)
        :param name:
        :param loss_f: lambda similarity_loss: return similarity_loss * my_special_layer
        :param calculate_org_similarity_loss Should the already calculated original similarity loss be passed (=True) to the loss function of y_true and y_pred?

        if calculate_org_similarity_loss: loss_f has the parameter "similarities_loss"
        if not calculate_org_similarity_loss: loss_f has the parameters "y_true", "y_pred"

        :return:
        """
        self._additional_grouping_similarity_losses.append({
            'name': name,
            'loss_f': loss_f,
            'calculate_org_similarity_loss': calculate_org_similarity_loss
        })

    def _register_additional_regularisation(self, layer, name, weight=1.0):
        if weight is not None:
            layer = Lambda(lambda x: x * weight)(layer)
        self._additional_regularisations.append({
            'layer': Activation('linear', name=name)(layer),
            'name': name
        })

    def _register_additional_embedding_comparison_regularisation(self, name, comparator_f, embeddings, weight=1.):
        self._additional_embedding_comparison_regularisations.append({
            'comparator_f': comparator_f, # comparator_f(e_i, e_j, cluster_i == cluster_j)
            'name': name,
            'embeddings': embeddings, # The ymust be ordered!
            'weight': weight,
        })

    def __build_comparisons_arrays(self, embeddings, comparator_f, weight):
        assert self.input_count == len(embeddings)

        cmp_eq = [] # All comparisons assuming the elements do equal
        cmp_ne = [] # All comparisons assuming the elements do not equal

        comparisons = 0
        for i_source in range(self.input_count):
            e_source = embeddings[i_source]

            # Should i_source be compared with itself?
            if self._include_self_comparison:
                target_range = range(i_source, self.input_count)
            else:
                target_range = range(i_source + 1, self.input_count)

            for i_target in target_range:
                e_target = embeddings[i_target]

                cmp_eq.append(comparator_f(e_source, e_target, 1.))
                cmp_ne.append(comparator_f(e_source, e_target, 0.))
                comparisons += 1

        # Create two long vectors for cmp_eq and cmp_ne
        cmp_eq = concat(cmp_eq, axis=1)
        cmp_ne = concat(cmp_ne, axis=1)

        # # Reshape and merge them
        # cmp_eq = Reshape((1, comparisons))(cmp_eq)
        # cmp_ne = Reshape((1, comparisons))(cmp_ne)
        #
        # # Concat both
        # cmp = concat([cmp_eq, cmp_ne], axis=1)

        cmp = concat([cmp_eq, cmp_ne], axis=1)

        # Weight the result
        cmp = cmp * weight

        # If this is not already the case: Convert the 'cmp' obj to a keras tensor (this command is a bit "dummy")
        cmp = Lambda(lambda x: cmp)(embeddings[0])

        return cmp

    def _get_cluster_count_possibilities(self):
        return self.data_provider.get_max_cluster_count() - self.data_provider.get_min_cluster_count() + 1

    def _register_plots(self):
        ClusterNN._register_plots(self)
        model_name = self._get_name('cluster_nn')

        # Add the cluster count accuracy plot
        def cluster_count_accuracy_plot(history, plt):
            x = list(history.get_epoch_indices())
            y_key = 'cluster_count_output_categorical_accuracy'
            y = history[y_key]
            y_val = history['val_{}'.format(y_key)]

            plt.plot(
                *filter_None(x, y),
                *filter_None(x, y_val),

                *filter_None(x, self.plot_sliding_window_average(y)),
                *filter_None(x, self.plot_sliding_window_average(y_val)),

                alpha=0.7,
                lw=0.5
            )
            plt.legend([
                'cluster count accuracy: training',
                'cluster count accuracy: validation',
                'cluster count accuracy: training AVG',
                'cluster count accuracy: validation AVG',
            ])
            plt.xlabel('iteration')
            plt.ylabel('cluster count accuracy')
            plt.grid(True)
        self._register_plot(model_name, cluster_count_accuracy_plot, 'loss', lambda history: 'cluster_count_output_categorical_accuracy' in history.keys())

        # Add the grouping accuracy plot
        def grouping_accuracy_plot(history, plt):
            x = list(history.get_epoch_indices())
            key_name = 'similarities_output_acc'
            if key_name not in history.keys():
                key_name = 'acc' # If only one fixed cluster count is used, the keyword changes (sorry, thats a bit ugly, keras!)
            y = history[key_name]
            y_val = history['val_{}'.format(key_name)]
            plt.plot(
                *filter_None(x, y),
                *filter_None(x, y_val),

                *filter_None(x, self.plot_sliding_window_average(y)),
                *filter_None(x, self.plot_sliding_window_average(y_val)),

                alpha=0.7,
                lw=0.5
            )
            plt.legend([
                'grouping accuracy: training',
                'grouping accuracy: validation',
                'grouping accuracy: training AVG',
                'grouping accuracy: validation AVG'
            ])
            plt.xlabel('iteration')
            plt.ylabel('grouping accuracy')
            plt.grid(True)
        self._register_plot(model_name, grouping_accuracy_plot, 'loss')

    def _build_y_data(self, inputs):
        cluster_counts = self.data_provider.get_cluster_counts()

        # Create output arrays
        if self.include_self_comparison:
            similarities_output_length = self.input_count * (self.input_count + 1) // 2
        else:
            similarities_output_length = self.input_count * (self.input_count - 1) // 2
        similarities_output = np.zeros((len(inputs), similarities_output_length), dtype=np.float32)
        cluster_count = np.zeros((len(inputs), len(cluster_counts)))

        for c in range(len(inputs)):
            current_inputs = inputs[c]['data']
            i = 0
            for i_source in range(self.input_count):
                ci_source = current_inputs[i_source][1]

                # Should i_source be compared with itself?
                if self._include_self_comparison:
                    target_range = range(i_source, self.input_count)
                else:
                    target_range = range(i_source + 1, self.input_count)

                for i_target in target_range:
                    ci_target = current_inputs[i_target][1]

                    similarities_output[c][i] = ci_source == ci_target
                    i += 1

            # # DEBUG:
            # if inputs[c]['cluster_count'] != 10:
            #     print(inputs[c])

            cluster_count[c][inputs[c]['cluster_count'] - cluster_counts[0]] = 1.

        y = {
            'similarities_output': similarities_output
        }

        # If there is more than one possible cluster count: Add the output for the cluster count
        if len(cluster_counts) > 1:
            y['cluster_count_output'] = cluster_count

        # If required: Add the additional similarity losses
        for additional_grouping_similarity_layer in self._additional_grouping_similarity_losses:
            y[additional_grouping_similarity_layer['name']] = similarities_output

        # If required: Add regularisations. They are also losses, but they do not use a "true" value; it is ignored
        for additional_regularisation in self._additional_regularisations:
            y[additional_regularisation['name']] = np.zeros((len(inputs), 1), dtype=np.float32)

        # If required add additional embedding comparison regularisations. They require the similarities output
        for additional_embedding_comparison_regularisation in self._additional_embedding_comparison_regularisations:
            y[additional_embedding_comparison_regularisation['name']] = similarities_output

        # # DEBUG output
        # c0 = np.    sum(similarities_output == 0)
        # c1 = np.sum(similarities_output == 1)
        # print("\nPrepared y data. Percentage of 0: {}; Percentage of 1: {}; Total values: {}".format(c0 / (c0 + c1), c1 / (c0 + c1), c0 + c1))

        return y

    def _build_loss_network(self, network_output, loss_output, additional_network_outputs):
        cluster_counts = self._get_cluster_counts()

        # Network output is a list of softmax distributions:
        # First in this list are all softmax distributions for the input i with first the softmax for k_min clusters,
        # then for k_min+1 clusters, etc.

        def get_softmax_dist(i_object, k):
            """
            Get the softmax distribution for the object i_object, assuming there are k clusters.
            :param i_object: The object index.
            :param k: The assumed cluster count.
            :return: The softmax distribution.
            """
            return additional_network_outputs['clusters']['input{}'.format(i_object)]['cluster{}'.format(k)]

        # Predefine some dot products which are required to compare softmax distributions
        k_dot_prod = {
            k: self._s_layer('softmax_dot_{}'.format(k), lambda name: Dot(axes=2, name=name)) for k in cluster_counts
        }

        # Compare all outputs with all outputs
        # softmax_dot_concat = self._s_layer('softmax_dot_concat', lambda name: Concatenate(name=name))
        softmax_dot_concat = self._s_layer('softmax_dot_concat', lambda name: concat_layer(name=name, input_count=len(cluster_counts)))
        softmax_dot_concat_reshaper = self._s_layer('softmax_dot_concat_reshaper', lambda name: Reshape((len(cluster_counts),), name=name))
        cluster_attention = self._s_layer('cluster_attention', lambda name: Dot(axes=1, name=name))
        n_cluster_output = additional_network_outputs['cluster_count_output']
        similarities = []
        for i_source in range(self.input_count):

            # Should i_source be compared with itself?
            if self._include_self_comparison:
                target_range = range(i_source, self.input_count)
            else:
                target_range = range(i_source + 1, self.input_count)

            for i_target in target_range:

                # For the source and the target, we compare now all softmax vectors with each other.
                # All these comparison results are then concated to one "long" vector.
                k_comparisons = []
                for k in cluster_counts:
                    k_comparisons.append(k_dot_prod[k]([
                        get_softmax_dist(i_source, k),
                        get_softmax_dist(i_target, k)
                    ]))

                comparisons_concat = softmax_dot_concat_reshaper(softmax_dot_concat(k_comparisons))

                # Do now some kind of "cluster attention" on the comparison values and add it to the similarities
                # list
                similarities.append(cluster_attention([comparisons_concat, n_cluster_output]))

        # Now all comparisons are stored in "similarities": Concate them and return the array
        # similarities_output = self._s_layer('similarities_output', lambda name: Concatenate(name=name), format_name=False)(similarities)
        similarities_output = self._s_layer('similarities_output', lambda name: concat_layer(name=name, input_count=len(similarities)), format_name=False)(similarities)
        loss_output.append(similarities_output)

        # Also add the cluster count output, but only if there is more than one possible cluster count and only if it is not fixed
        if len(cluster_counts) > 1:
            loss_output.append(n_cluster_output)

        # Are there any additional similarity losses defined? If yes: add them to the output; they all get the similarities output.
        # We need to create some wrapper layers to allow this. Just use "empty" activations
        for additional_grouping_similarity_loss in self._additional_grouping_similarity_losses:
            loss_output.append(Activation('linear', name=additional_grouping_similarity_loss['name'])(similarities_output))

        # Are there any additional regularisations defined? If yes: add them ass loss
        for additional_regularisation in self._additional_regularisations:
            loss_output.append(additional_regularisation['layer'])

        # If required add additional embedding comparison regularisations
        for additional_embedding_comparison_regularisation in self._additional_embedding_comparison_regularisations:
            comparisons = self.__build_comparisons_arrays(
                additional_embedding_comparison_regularisation['embeddings'],
                additional_embedding_comparison_regularisation['comparator_f'],
                additional_embedding_comparison_regularisation['weight']
            )
            loss_output.append(Activation(
                'linear',
                name=additional_embedding_comparison_regularisation['name']
            )(comparisons))

        return True

    def get_class_weights(self):
        if not self._weighted_classes:
            return None
        if self._class_weights_approximator == 'simple_approximation':
            return self.__get_simple_approximation_class_weights()
        elif self._class_weights_approximator == 'stochastic':
            return self.__get_stochastic_class_weights()
        else:
            raise ValueError()

    def __get_simple_approximation_class_weights(self):

        # A function that calculates the expected ones in the similarities_output for a given cluster count
        # (from a mathematical point of view this function is not completely correct, but it is a good approximation)
        def expected_ones(cluster_count):
            expected_cluster_size = self.input_count / cluster_count
            if self.include_self_comparison:
                expected_connections_per_cluster = expected_cluster_size * (expected_cluster_size + 1) / 2
            else:
                expected_connections_per_cluster = expected_cluster_size * (expected_cluster_size - 1) / 2
            return cluster_count * expected_connections_per_cluster

        # Calculate the expected ones over all possible cluster counts
        cluster_counts_distribution = self.data_provider.get_cluster_counts_distribution()
        total_expected_ones = sum([
            expected_ones(cluster_count) * p for cluster_count, p in cluster_counts_distribution.items()
        ])

        # And also the expected zeros
        if self._include_self_comparison:
            total_outputs = self.input_count * (self.input_count + 1) / 2
        else:
            total_outputs = self.input_count * (self.input_count - 1) / 2
        total_expected_zeros = total_outputs - total_expected_ones

        # Create now the class weights, based on these equations:
        # w_0 = weight for zeros
        # w_1 = weights for ones
        # c_0 = total_expected_zeros
        # c_1 = total_expected_ones
        #
        # w_0 * c_0 = w_0 * c_0
        # w_0 + w_1 = weights_count = 1
        #
        # If we solve these two equations, we get:
        # w_0 = c_1 / (c_0 + c_1)
        # w_1 = c_0 / (c_0 + c_1) = 1 -  w_0
        w0 = total_expected_ones / (total_expected_zeros + total_expected_ones)
        w1 = 1 - w0

        print("Calculated class weights: w0={}, w1={}".format(w0, w1))
        return w0, w1

    def __get_stochastic_class_weights(self):

        # The required settings: The difference between the upper and lower limit of the 95% confidence interval must
        # be smaller than 0.005
        confidence = 0.95
        max_diff = 0.005

        def sample_data():

            # Generate data
            data, _, _ = self._get_data('train')
            _, y = self._build_Xy_data(data)

            # Get the y-data and calculate the expected percentage of '0s' in the similarities output
            similarities_output = y['similarities_output']
            return [
                np.mean(similarities_output[i]) for i in range(len(data))
            ]

        t_start = time()
        data = []
        print("Start to calculate the expected stochastic class weights...")
        while True:
            print("Sample some data...")
            data += sample_data()
            print("Sample data count: {}".format(len(data)))

            if min(data) == max(data):
                print("All samples similarity probabilities do equal. Cannot calculate confidence interval.")
                if len(data) >= 1000:
                    print("Assuming all records are always equal: Just use the current (constant) value as mean")
                    lower_bound = upper_bound = data[0]
                else:
                    lower_bound = upper_bound = None
            else:
                lower_bound, upper_bound = mean_confidence_interval(data, confidence=confidence)
            if lower_bound is not None and upper_bound is not None:
                interval_size = upper_bound - lower_bound
            else:
                interval_size = float('NaN')
            print("{}% confidence interval size (must be smaller than {}): {}".format(confidence * 100, max_diff, interval_size))

            if interval_size <= max_diff:
                mean = (upper_bound + lower_bound) / 2
                print("Interval is ok. Mean: {}".format(mean))
                break
            else:
                print("Interval is larger than {}. Sample more data...".format(max_diff))
        t_end = time()
        print("Required {} seconds...".format(t_end - t_start))

        expected_ones_percentage = mean
        expected_zeros_percentage = 1 - mean

        # Calculate the weights
        w0 = expected_ones_percentage
        w1 = expected_zeros_percentage

        print("Calculated class weights: w0={}, w1={}".format(w0, w1))

        return w0, w1

    def _get_keras_loss(self):
        if self._similarities_loss_f is not None:
            print("Use a custom similarities loss function")
            similarities_loss = self._similarities_loss_f
        elif self.weighted_classes:

            # Calculate the class weights
            w0, w1 = self.get_class_weights()
            print("Calculated class weights: w0={}, w1={}".format(w0, w1))

            # If required: Reweight them
            if self._class_weights_post_processing_f is not None:
                print("Post-process the calculated weights (e.g. use the sqrt / log / etc.)")

                # Check the count of parameters of the post-processing function: If 1 argument is used, then
                # execute it for both weights, if it has two arguments input both weights at once
                arg_count = len(signature(self._class_weights_post_processing_f).parameters)
                if arg_count == 1:
                    w0 = self._class_weights_post_processing_f(w0)
                    w1 = self._class_weights_post_processing_f(w1)
                elif arg_count == 2:
                    w0, w1 = self._class_weights_post_processing_f(w0, w1)
                else:
                    raise Exception("Invalid argument count for self._class_weights_post_processing_f: 1 or 2 arguments are required, but the current function has {} arguments".format(arg_count))
                print("Post-processed weights: w0={}, w1={}".format(w0, w1))

            # Normalize the weights if required: The sum should be equal to 2
            if self._normalize_class_weights:
                print("Normalize the class weights")
                s = w0 + w1
                w0 = 2 * w0 / s
                w1 = 2 * w1 / s
                print("Normalized weights: w0={}, w1={}".format(w0, w1))

            print("Final calculated weights: w0={}, w1={}".format(w0, w1))
            similarities_loss = create_weighted_binary_crossentropy(w0, w1)
        else:
            print("Use the standard non-weighted binary crossentropy loss")
            similarities_loss = binary_crossentropy # 'binary_crossentropy'
        loss = {}
        if self._use_similarities_loss:
            loss['similarities_output'] = similarities_loss
        if self._use_cluster_count_loss and len(self._get_cluster_counts()) > 1:
            loss['cluster_count_output'] = self._cluster_n_output_loss

        # Register all additional similarity losses
        for additional_grouping_similarity_loss in self._additional_grouping_similarity_losses:
            name = additional_grouping_similarity_loss['name']
            loss_f = additional_grouping_similarity_loss['loss_f']
            if additional_grouping_similarity_loss['calculate_org_similarity_loss']:
                loss[name] = lambda y_true, y_pred, loss_f=loss_f: loss_f(similarities_loss(y_true, y_pred))
            else:
                loss[name] = lambda y_true, y_pred, loss_f=loss_f: loss_f(y_true, y_pred)

        # Register all regularisations
        if len(self._additional_regularisations) > 0:
            regularisation_loss = lambda y_true, y_pred: y_pred
            for additional_regularisation in self._additional_regularisations:
                loss[additional_regularisation['name']] = regularisation_loss

        # Register all comparison regularisations
        if len(self._additional_embedding_comparison_regularisations):

            # The encoding of the values is a bit hacky:
            # y_pred contains all "true" comparisons and then all "false" comparisons.
            # We have to calculate the size of cmp_eq and cmp_ne
            if self.include_self_comparison:
                n = self.input_count * (self.input_count + 1) // 2
            else:
                n = self.input_count * (self.input_count - 1) // 2

            def embedding_comparison_regularisation_loss(y_true, y_pred):
                cmp_eq = y_pred[:, :n]
                cmp_ne = y_pred[:, n:]
                return K.mean(
                    y_true * cmp_eq + (1 - y_true) * cmp_ne,
                    axis=-1
                )
            for additional_embedding_comparison_regularisation in self._additional_embedding_comparison_regularisations:
                loss[additional_embedding_comparison_regularisation['name']] = embedding_comparison_regularisation_loss

        return loss

    def _get_keras_metrics(self):
        metrics = {
            'similarities_output': 'accuracy'
        }
        if len(self.data_provider.get_cluster_counts()) > 1:
            metrics['cluster_count_output'] = 'categorical_accuracy'
        return metrics
