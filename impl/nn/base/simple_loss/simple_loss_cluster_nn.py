import numpy as np

from keras.layers import Concatenate, Dot, Reshape

from core.nn.cluster_nn import ClusterNN
from core.nn.helper import filter_None, concat_layer


class SimpleLossClusterNN(ClusterNN):
    def __init__(self, data_provider, input_count, embedding_nn=None):
        ClusterNN.__init__(self, data_provider, input_count, embedding_nn)

        # For the loss all n outputs are compared with each other. More exactly:
        # n(n-1)/2 comparisons are required. If the following setting is True, then
        # each element is also comapred with itself. But why? This forces the network
        # to produce distributions with one large value.
        self._include_self_comparison = True

    @property
    def include_self_comparison(self):
        return self._include_self_comparison

    @include_self_comparison.setter
    def include_self_comparison(self, include_self_comparison):
        self._include_self_comparison = include_self_comparison

    def _get_cluster_count_possibilities(self):
        return self.data_provider.get_max_cluster_count() - self.data_provider.get_min_cluster_count() + 1

    def _register_plots(self):
        ClusterNN._register_plots(self)
        model_name = self._get_name('cluster_nn')

        # Add the cluster count accuracy plot
        def cluster_count_accuracy_plot(history, plt):
            x = list(history.get_epoch_indices())
            plt.plot(
                *filter_None(x, history['cluster_count_output_categorical_accuracy']),
                *filter_None(x, history['val_cluster_count_output_categorical_accuracy']),
                alpha=0.7,
                lw=0.5
            )
            plt.legend(['cluster count accuracy: training', 'cluster count accuracy: validation'])
            plt.xlabel('iteration')
            plt.ylabel('cluster count accuracy')
            plt.grid(True)
        self._register_plot(model_name, cluster_count_accuracy_plot)

        # Add the grouping accuracy plot
        def grouping_accuracy_plot(history, plt):
            x = list(history.get_epoch_indices())
            key_name = 'similarities_output_acc'
            if key_name not in history.keys():
                key_name = 'acc' # If only one fixed cluster count is used, the keyword changes (sorry, thats a bit ugly, keras!)
            plt.plot(
                *filter_None(x, history[key_name]),
                *filter_None(x, history['val_{}'.format(key_name)]),
                alpha=0.7,
                lw=0.5
            )
            plt.legend(['grouping accuracy: training', 'grouping accuracy: validation'])
            plt.xlabel('iteration')
            plt.ylabel('grouping accuracy')
            plt.grid(True)
        self._register_plot(model_name, grouping_accuracy_plot)

    def _build_y_data(self, inputs):
        cluster_counts = self.data_provider.get_cluster_counts()

        # Create output arrays
        similarities_output = np.zeros((len(inputs), self.input_count * (self.input_count + 1) // 2), dtype=np.float32)
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

        return y

    def _build_loss_network(self, network_output, loss_output, additional_network_outputs):
        cluster_counts = self.data_provider.get_cluster_counts()

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

        # Also add the cluster count output, but only if there is more than one possible cluster count
        if len(cluster_counts) > 1:
            loss_output.append(n_cluster_output)

        return True

    def _get_keras_loss(self):
        loss = {
            'similarities_output': 'binary_crossentropy'
        }
        if len(self.data_provider.get_cluster_counts()) > 1:
            loss['cluster_count_output'] = 'categorical_crossentropy'
        return loss

    def _get_keras_metrics(self):
        metrics = {
            'similarities_output': 'accuracy'
        }
        if len(self.data_provider.get_cluster_counts()) > 1:
            metrics['cluster_count_output'] = 'categorical_accuracy'
        return metrics
