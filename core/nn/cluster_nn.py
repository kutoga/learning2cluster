from itertools import chain, combinations
from random import Random
from os import path
from time import time
import collections

import numpy as np

from sklearn import metrics

from termcolor import colored

from keras.models import Model
from keras.layers import Input, Activation, Reshape, TimeDistributed

from core.nn.base_nn import BaseNN
from core.nn.history import History
from core.nn.helper import filter_None, AlignedTextTable, np_show_complete_array, get_caller, concat_layer, slice_layer
from core.event import Event
from core.helper import try_makedirs, index_of

from core.nn.external.purity import purity_score
from core.nn.misc.MR import misclassification_rate_BV01
from core.nn.misc.BBN import BBN

class ClusterNN(BaseNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, seed=None, create_metrics_plot=True,
                 include_input_count_in_name=True, validation_data_count=None, early_stopping_itrs=None,
                 use_hints_for_training=True):
        super().__init__(
            name='NN_[CLASS]_I{}'.format(input_count) if include_input_count_in_name else 'NN_[CLASS]'
        )
        self._rand = Random(seed)
        self._data_provider = data_provider
        self._input_count = input_count
        self._embedding_nn = embedding_nn
        self._create_metrics_plot = create_metrics_plot

        self._optimizer = 'adadelta'
        self._minibatch_size = 100

        self._early_stopping_itrs = early_stopping_itrs # None => no early stopping is used

        self._train_history = History()
        self._validate_every_nth_epoch = 10
        self._validation_data_count = validation_data_count # None => use the batch size

        self._model_prediction = None
        self._model_training = None

        # TBD: Currently ignored
        self._model_discriminator_real = None
        self._model_discriminator_fake = None

        # Some events
        self.event_training_iteration_before = Event()
        self.event_training_iteration_after = Event()
        self.event_new_best_validation_loss = Event()
        self.event_new_best_training_loss = Event()
        self.event_early_stopped = Event()

        # The cluster count function. It may be overwritten (default=random)
        self._f_cluster_count = None

        # Debug outputs
        self._prediction_debug_outputs = None

        # Additional prediction outputs
        self._additional_prediction_outputs = None

        # Evaluation metrics: Initialize variables and register the default metrics
        self._evaluation_metrics = {}
        self.__register_default_evaluation_metrics()
        if self._create_metrics_plot:
            self.__register_evalution_metrics_plots()

        # Normalize network input
        self._normalize_network_input = False # Compatibility

        # Different losses may be weighted differently: Default weights are just 1
        self._loss_weights = {}

        # The hints input for the network
        self.__nw_clustering_hint = None

        # Use hints for the training? (if hints are available; otherwise this setting has no effect)
        self.__use_hints_for_training = use_hints_for_training

        # Define some additional build config value; They do not have a pre-defined meaning, but they allow it to pass
        # some information from the build_network call to the code that creates the neural network
        self._additional_build_config = {}

        # Is the network already built?
        self._network_built = False

    @property
    def is_network_built(self):
        return self._network_built

    @property
    def normalize_network_input(self):
        return self._normalize_network_input

    @normalize_network_input.setter
    def normalize_network_input(self, normalize_network_input):
        self._normalize_network_input = normalize_network_input

    @property
    def validate_every_nth_epoch(self):
        return self._validate_every_nth_epoch

    @validate_every_nth_epoch.setter
    def validate_every_nth_epoch(self, validate_every_nth_epoch):
        self._validate_every_nth_epoch = validate_every_nth_epoch

    @property
    def data_provider(self):
        return self._data_provider

    @property
    def input_count(self):
        return self._input_count

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, minibatch_size):
        self._minibatch_size = minibatch_size

    @property
    def f_cluster_count(self):
        return self._f_cluster_count

    @f_cluster_count.setter
    def f_cluster_count(self, f_cluster_count):
        self._f_cluster_count = f_cluster_count

    @property
    def validation_data_count(self):
        return self._validation_data_count

    @validation_data_count.setter
    def validation_data_count(self, validation_data_count):
        self._validation_data_count = validation_data_count

    @property
    def early_stopping_iterations(self):
        return self._early_stopping_itrs

    @early_stopping_iterations.setter
    def early_stopping_iterations(self, early_stopping_itrs):
        self._early_stopping_itrs = early_stopping_itrs

    @property
    def use_hints_for_training(self):
        return self.__use_hints_for_training

    @use_hints_for_training.setter
    def use_hints_for_training(self, use_hints_for_training):
        self.__use_hints_for_training = use_hints_for_training

    def set_loss_weight(self, loss_name, weight=None):
        self._loss_weights[loss_name] = weight

    def reset_loss_weight(self, loss_name):
        if loss_name in self._loss_weights:
            del self._loss_weights[loss_name]

    def reset_loss_weights(self):
        self._loss_weights = {}

    def register_evaluation_metric(self, name, f_metric):
        self._evaluation_metrics[name] = f_metric

    def __register_default_evaluation_metrics(self):

        # These metrics are described here: http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
        for name, f_metric in [
            ('adjusted_rand_score', metrics.adjusted_rand_score),
            ('adjusted_mutual_info_score', metrics.adjusted_mutual_info_score),
            ('normalized_mutual_info_score', metrics.normalized_mutual_info_score),
            ('mutual_info_score', metrics.mutual_info_score),
            ('homogeneity_score', metrics.homogeneity_score),
            ('completeness_score', metrics.completeness_score),
            ('v_measure_score', metrics.v_measure_score),
            ('fowlkes_mallows_score', metrics.fowlkes_mallows_score),
            ('purity_score', purity_score),

            # BV01 = Beta Version 01
            # The reason for this version number is that this metric may change
            # in future (currently it is (probably) not optiomal and just an upper bound).
            ('misclassification_rate_BV01', misclassification_rate_BV01),

            # Register the un-normalized and the normalized BBN metric. For both Q=0 is used.
            ('bbn_q0', lambda y_true, y_pred: BBN(y_true, y_pred, Q=0, normalize=False)),
            ('bbn_q0_normalized', lambda y_true, y_pred: BBN(y_true, y_pred, Q=0, normalize=True))
        ]:
            self.register_evaluation_metric(name, f_metric)

    def __register_evalution_metrics_plots(self):
        metrics_plot_name = 'metrics'
        model_name = self._get_name('cluster_nn')

        # Clean up
        self._clear_registered_plots(metrics_plot_name)

        # Register all plots
        for metric in sorted(self._evaluation_metrics.keys()):

            # Add the loss plot
            def metric_plot(history, plt, metric=metric):
                x = list(history.get_epoch_indices())
                y = history['metric_{}'.format(metric)]
                x_p, y_p = filter_None(x, y)
                y_min = min(y_p)
                y_max = max(y_p)
                plt.plot(
                    x_p, y_p,
                    *filter_None(x, self.plot_sliding_window_average(y)),

                    alpha=0.7,
                    lw=0.5
                )
                plt.legend([
                    '{}: validation'.format(metric),
                    '{}: validation AVG'.format(metric)
                ])
                if y_max <= 1 and y_min >= -1:
                    lower_limit = -1 if y_min < 0 else 0
                    upper_limit = 1
                    plt.ylim((lower_limit, upper_limit))
                plt.xlabel('iteration')
                plt.ylabel('score')
                plt.grid(True)
            self._register_plot(model_name, metric_plot, metrics_plot_name)

    def _uses_embedding_layer(self):
        return self._embedding_nn is not None

    def calculate_embeddings(self, x):

        # Handle list inputs
        if isinstance(x, list):
            return list(map(self.calculate_embeddings, x))

        # Is an embedding layer used?
        if not self._uses_embedding_layer():
            return x

        # Predict the embeddings
        x = np.asarray([x])
        embedding = self._embedding_nn.model.predict(x)[0]

        return embedding

    def _get_embedding(self, layer, time_distributed=False, layer_base_name='embedding_preprocessor'):

        # If a list of layers is given: Return the embedding for each layer
        if isinstance(layer, list):
            if time_distributed:

                # Simple case: if there is no embedding-network, just return the input
                if self._embedding_nn is None:
                    return layer

                # Use the "time-distributed" mode: Merge all inputs together, create the Embeddings and then split them
                shape = layer[0]._keras_shape[1:]

                # Prepare the required layers
                reshape_layer = self._s_layer(
                    '{}_concat_init_reshape'.format(layer_base_name),
                    lambda name: Reshape((1,) + shape, name=name)
                )
                merge_layer = self._s_layer(
                    '{}_concat'.format(layer_base_name),
                    lambda name: concat_layer(axis=1, name=name, input_count=len(layer))
                )

                # Merge all input together
                merged = merge_layer([
                    reshape_layer(l) for l in layer
                ])

                # Create now the embeddings
                embeddings = TimeDistributed(self._embedding_nn.model)(merged)

                # Slice them all: We want to split the result
                shape = embeddings._keras_shape[1:]
                reshape_layer = self._s_layer(
                    '{}_concat_final_reshape'.format(layer_base_name),
                    lambda name: Reshape(shape[1:], name=name)
                )
                embeddings_list = [
                    reshape_layer(
                        self._s_layer('{}_slice_{}'.format(layer_base_name, i), lambda name: slice_layer(embeddings, i, name))
                    ) for i in range(len(layer))
                ]

                # Thats it
                return embeddings_list

            else:

                # Create the embedding for each input independent of each other
                return [self._get_embedding(l) for l in layer]

        # Return the embedding for the given layer
        if self._embedding_nn is None:
            return layer
        embedding_model = self._embedding_nn.model
        return embedding_model(layer)

    def _register_plots(self):
        BaseNN._register_plots(self)
        model_name = self._get_name('cluster_nn')

        # Add the loss plot
        def loss_plot(history, plt):
            x = list(history.get_epoch_indices())
            y = history['loss']
            y_val = history['val_loss']
            plt.plot(
                *filter_None(x, y),
                *filter_None(x, y_val),

                *filter_None(x, self.plot_sliding_window_average(y)),
                *filter_None(x, self.plot_sliding_window_average(y_val)),

                alpha=0.7,
                lw=0.5
            )
            plt.legend([
                'loss: training',
                'loss: validation',
                'loss: training AVG',
                'loss: validation AVG'
            ])
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.grid(True)
        self._register_plot(model_name, loss_plot, 'loss')

        # Add the new best loss plot
        def best_loss_plot(history, plt):
            x = list(history.get_epoch_indices())

            # Create a function to get the best loss values (1 or 0)
            def get_best_loss_y_values(losses):
                best_loss = None
                y = []
                for loss in losses:
                    if loss is None:
                        y.append(None)
                    elif best_loss is None or loss < best_loss:
                        best_loss = loss
                        y.append(1)
                    else:
                        y.append(0)
                return y

            plt.plot(
                *filter_None(x, get_best_loss_y_values(history['loss'])),
                *filter_None(x, get_best_loss_y_values(history['val_loss'])),
                alpha=0.7
            )
            plt.legend(['new best loss: training', 'new best loss: validation'])
            plt.xlabel('iteration')
            plt.ylabel('new best loss')
            plt.grid(True)
        self._register_plot(model_name, best_loss_plot, 'loss')

    def _build_network(self, network_input, network_output, additional_network_outputs):
        return None

    def _build_loss_network(self, network_output, loss_output, additional_network_outputs):
        return None

    def _reweight_loss(self, loss, weight):
        return lambda y_true, y_pred: weight * loss(y_true, y_pred)

    def _get_weighted_keras_loss(self):
        losses = self._get_keras_loss()
        loss_weigths = self._loss_weights
        for loss_name in sorted(losses.keys()):
            if loss_name in loss_weigths and loss_weigths[loss_name] is not None:
                weight = loss_weigths[loss_name]
                print("Reweight the loss {} with the factor {}".format(loss_name, weight))
                losses[loss_name] = self._reweight_loss(losses[loss_name], weight)
        return losses

    def _get_keras_loss(self):
        return None

    def _get_keras_metrics(self):
        return {}

    def _build_discriminator_network(self, network_output, discriminator_output):
        return None

    def _get_discriminator_weight(self):
        return 0.

    def _get_loss_weight(self):
        return 1.

    def _build_y_data(self, inputs):
        pass

    def __build_X_data(self, inputs):
        data_shape = self.data_provider.get_data_shape()
        X = [
            np.zeros((len(inputs),) + data_shape, dtype=np.float32) for i in range(self._input_count)
        ]
        for c in range(len(inputs)):
            current_inputs = inputs[c]['data']
            assert len(current_inputs) == len(X)
            for i in range(len(current_inputs)):
                X[i][c] = self._normalize_array_if_required(current_inputs[i][0])

        # Append hints
        X.append(self.__hints_to_np_arr(
            list(map(lambda inp: inp['hints'], inputs)),
            len(inputs)
        ))

        return X

    def _build_Xy_data(self, data, hints=None):

        # Prepare the data:
        # [
        #     {
        #         'cluster_count': 5,
        #         'data': [datapoint1, datapoint2, ...]
        #     },
        #     {
        #         'cluster_count': 3,
        #         'data': [datapointX, datapointY, ...]
        #     }
        # ]
        inputs = []
        for c in range(len(data)):
            clusters = data[c]

            # Create a permutation for the cluster indices
            cluster_indices = list(range(len(clusters)))
            self._rand.shuffle(cluster_indices)

            # Collect all inputs
            current_inputs = list(chain.from_iterable(
                map(
                    lambda ci: list(map(lambda d: (d, cluster_indices[ci]), clusters[ci])),
                    range(len(clusters))
                )
            ))

            if len(current_inputs) != self._input_count:
                print("Error: Invalid input count (expected {}, but got {})".format(self._input_count, len(current_inputs)))

            # Shuffle all inputs
            self._rand.shuffle(current_inputs)

            # Generate the hints (if some hints are given)
            current_hints = []
            if hints is not None and hints[c] is not None:
                current_x_inputs = list(map(lambda x: x[0], current_inputs))
                for hint in hints[c]:
                    current_hints.append(list(map(
                        lambda x: index_of(current_x_inputs, x[0]),
                        hint
                    )))

            # That's it for the current cluster collection:)
            inputs.append({
                'cluster_count': len(clusters),
                'data': current_inputs,
                'hints': current_hints
            })

        return self.__build_X_data(inputs), self._build_y_data(inputs)

    def __build_elements_inputs(self):
        return [
            self._s_layer('input_{}'.format(i), lambda name: Input(shape=self.data_provider.get_data_shape(), name=name))
            for i in range(self._input_count)
        ]

    def __get_last_epoch(self):
        return self._get_history(self._model_training).length()

    def _get_data(self, data_type='train', dummy_data=False, cluster_collection_count=None, *args):
        if cluster_collection_count is None:
            cluster_collection_count = self._minibatch_size
        clusters, additional_obj_info, hints = self._data_provider.get_data(self._input_count, cluster_collection_count, data_type=data_type, dummy_data=dummy_data, cluster_count_f=self._f_cluster_count, *args)

        return clusters, additional_obj_info, hints

        # # TODO: Use / return hints
        #
        # # If required also return the object infos
        # if return_additional_obj_infos:
        #     return clusters, additional_obj_info
        # else:
        #     return clusters

    def _get_cluster_counts(self):
        return list(self._data_provider.get_target_cluster_counts())

    def _normalize_array(self, arr):
        arr = arr - np.mean(arr)
        arr = arr / (np.std(arr) + 1e-8)
        return arr

    def _normalize_array_if_required(self, arr):
        if self._normalize_network_input:
            arr = self._normalize_array(arr)
        return arr

    def __train_iteration(self, dummy_train=False):
        if self._model_training is None:
            raise Exception("No training model is defined (it has to be built with 'build_networks(build_training_model=True)'")

        self.event_training_iteration_before.fire(nth=self.__get_last_epoch())
        do_validation = (self.__get_last_epoch() + 1) % self._validate_every_nth_epoch == 0
        cluster_counts = self._get_cluster_counts()  #list(self._data_provider.get_cluster_counts())

        # Generate training data
        t_start_data_gen_time = time()
        train_data, _, train_hints = self._get_data('train', dummy_data=dummy_train)
        X_train, y_train = self._build_Xy_data(train_data, train_hints if self.__use_hints_for_training else None)

        # If required: Generate validation data
        if do_validation:
            valid_data, _, valid_hints = self._get_data('valid', dummy_data=dummy_train, cluster_collection_count=self._validation_data_count)
            validation_data = self._build_Xy_data(valid_data, valid_hints)
        else:
            valid_data = None
            validation_data = None
        t_end_data_gen_time = time()
        t_data_gen = t_end_data_gen_time - t_start_data_gen_time

        # Get some information to print
        history = self._get_history(self._model_training)
        best_valid_loss_itr = history.get_min_index('val_loss')
        best_valid_loss = None
        if best_valid_loss_itr is not None:
            best_valid_loss = history['val_loss'][best_valid_loss_itr]
            best_valid_loss_itr += 1
        best_train_loss_itr = history.get_min_index('loss')
        best_train_loss = None
        if best_train_loss_itr is not None:
            best_train_loss = history['loss'][best_train_loss_itr]
            best_train_loss_itr += 1

        # Print an info message
        print(
            '\nIteration ' + colored(str(self.__get_last_epoch() + 1), 'blue', attrs=['bold']) +
            '\n(incl. valid data: {}; best valid loss (iter. {}): {:0.6}; best train loss (iter. {}): {:0.6}; ' \
            'training time: {:0.9}s; data gen. time: {:0.9}s)'.format(
                do_validation,
                best_valid_loss_itr, best_valid_loss or float('nan'),
                best_train_loss_itr, best_train_loss or float('nan'),
                float(sum(filter(lambda t: t is not None, history['time_s']))),
                t_data_gen
        ))

        # Do the training
        t_start = time()
        self._model_training.fit(
            X_train, y_train, epochs=1, verbose=0, shuffle=True, batch_size=self._minibatch_size,
            validation_data=validation_data
        )
        t_end = time()
        t_train = t_end - t_start

        # Store the history
        history.load_keras_history(self._model_training.history.history)

        # Also store the required training time
        history.get_or_create_item('time_s')[-1] = t_train

        # Is validation data used? If yes: Then we have to calculate the metrics
        if do_validation:
            valid_metrics, valid_prediction = self.evaluate_metrics(valid_data, return_prediction=True)
            current_metrics = {metric: [] for metric in valid_metrics[0].keys()}
            for c_i in range(len(valid_data)):

                # Get the most probable cluster count
                most_probable_cluster_count = np.argmax(valid_prediction[c_i]['cluster_count']) + cluster_counts[0]

                # Add all metrics to current_metrics
                for metric in valid_metrics[c_i].keys():
                    current_metrics[metric].append(valid_metrics[c_i][metric][most_probable_cluster_count])

            # Average all metrics and add them to the history
            for metric in current_metrics.keys():
                history.get_or_create_item('metric_{}'.format(metric))[-1] = np.mean(current_metrics[metric])

        # A helper function that may be used to check if the current value is the "best" available value
        def latest_is_minimum_value(key):
            if history[key][-1] is None:
                return False
            if history.length() == 1:
                return True
            min_v = min(filter(lambda v: v is not None, history[key][0:-1]), default=None)
            if min_v is None:
                return True
            return min_v > history[key][-1]

        # Create a nice output for the command line:
        # [Training Loss: ][bold train-color loss], [other training values: ][bold train-color loss]
        # [Validation Loss: ][bold valid-color loss], [other validation values: ][bold valid-color loss]
        # Required time: [bold time_s]
        #
        # Align the training and the validation loss values
        # TODO: Put this code in a function
        tbl = AlignedTextTable(add_initial_row=False)
        tbl_metrics = AlignedTextTable(add_initial_row=True)
        tbl_metrics_avg = AlignedTextTable(add_initial_row=True)
        tbl_metrics_best = AlignedTextTable(add_initial_row=True)
        tbl_metrics_best_itr = self.__get_last_epoch() if latest_is_minimum_value("val_loss") else best_valid_loss_itr
        if tbl_metrics_best_itr is not None:
            tbl_metrics_best_itr -= 1
        metrics_avg_n = 20
        train = {}
        valid = {}
        others = {}
        float_value_format = '{0:.6}'
        for key in sorted(history.keys()):
            value = history[key][-1]
            if value is None:
                continue
            if key == "time_s":
                others[key] = value
            elif key.startswith("val_"):
                valid[key] = value
            elif key.startswith("metric_"):

                # Add the metrics value
                tbl_metrics.add_cell(key)
                tbl_metrics.add_cell(colored(float_value_format.format(value), attrs=['bold']))

                # Add the average over the last metrics_avg_n values
                tbl_metrics_avg.add_cell("{}_avg{}".format(key, metrics_avg_n))
                tbl_metrics_avg.add_cell(colored(float_value_format.format(history.aggregate_over_latest_values(key, f_aggregate=np.mean, n=metrics_avg_n)), attrs=['bold']))

                # Add the metrics for the best run
                tbl_metrics_best.add_cell("{}_best".format(key))
                tbl_metrics_best.add_cell(colored(float_value_format.format(history[key][tbl_metrics_best_itr]), attrs=['bold']))

            else:
                train[key] = value
        for row in filter(lambda d: len(d) > 0, [train, valid, others]):
            tbl.new_row()
            used_keys = set()
            def try_add(key, color=None, bold=False):
                if key in row:
                    attrs = []
                    if bold:
                        attrs.append('bold')
                    used_keys.add(key)
                    value = row[key]
                    if value is not None:
                        tbl.add_cell("{}:".format(key))
                        tbl.add_cell(colored(float_value_format.format(value), color=color, attrs=attrs))
                    del row[key]
            def try_add_all(key_contains, color=None, bold=False):
                for key in sorted(list(filter(lambda k: key_contains in k, row.keys()))):
                    try_add(key, color, bold)
            try_add("loss", color="green" if latest_is_minimum_value("loss") else "yellow", bold=True)
            try_add("val_loss", color="green" if latest_is_minimum_value("val_loss") else "yellow", bold=True)
            try_add_all("accuracy", color="blue", bold=True)
            try_add_all("acc", color="blue", bold=True)
            for k in list(sorted(row.keys())):
                try_add(k, bold=True)
        tbl.print_str()

        tbl_metrics = AlignedTextTable.merge(tbl_metrics, tbl_metrics_avg, tbl_metrics_best)
        tbl_metrics.print_str()

        # Old print code:
        # for hist_key in sorted(history.keys()):
        #     value = history[hist_key][-1]
        #     if value is not None:
        #         print('{}: {:.5f} '.format(hist_key, value), end='')
        # print()

        if latest_is_minimum_value('loss'):
            self.event_new_best_training_loss.fire(history, history['loss'][-1])
        if latest_is_minimum_value('val_loss'):
            self.event_new_best_validation_loss.fire(history, history['val_loss'][-1])

        self.event_training_iteration_after.fire(history, nth=self.__get_last_epoch())

    def train(self, iterations=1):
        early_stopped = False
        iterations_done = 0
        for i in range(iterations):

            # Is early stopping enabled?
            if self._early_stopping_itrs is not None:
                history = self._get_history(self._model_training)
                best_valid_loss_itr = history.get_min_index('val_loss')
                latest_itr = self.__get_last_epoch()
                if not (latest_itr is None or best_valid_loss_itr is None):
                    if latest_itr - best_valid_loss_itr >= self._early_stopping_itrs:
                        print("Early stopping (after {} iterations without any improvement on the validation data)".format(self._early_stopping_itrs))
                        early_stopped = True
                        break

            # Do a training iteration
            self.__train_iteration()
            iterations_done += 1

        # If required: Fire the early stopped event
        if early_stopped:
            self.event_early_stopped.fire(iterations_done)

        return early_stopped

    def data_to_cluster_indices(self, data, shuffle_indices=None):
        if len(data[0]) > 0 and isinstance(data[0][0], list):
            return list(map(
                lambda i: self.data_to_cluster_indices(data[i], None if shuffle_indices is None else shuffle_indices[i]),
                range(len(data))
            ))
        if len(data[0]) == 0:
            print("There are empty clusters in the input. This may produce detection problems.")

        cluster_indices = list(chain.from_iterable(map(
            lambda i: [i] * len(data[i]),
            range(len(data))
        )))

        # Shuffle if required
        if shuffle_indices:
            cluster_indices = [cluster_indices[i] for i in shuffle_indices]

        return cluster_indices

    def evaluate_metrics(self, data, return_prediction=False, shuffle_data=True):
        # TBD: Convert data and call "evaluate_metrics_from_prediction(self, X, cluster_indices, prediction):"

        data_X, data_hints, data_idx,  = self._data_provider.convert_data_to_prediction_X(data, shuffle=shuffle_data)
        prediction = self.predict(data_X, data_hints)

        cluster_indices = self.data_to_cluster_indices(data, data_idx)
        metrics = self.evaluate_metrics_from_prediction(prediction, cluster_indices)

        if return_prediction:
            return metrics, prediction
        else:
            return metrics


    def evaluate_metrics_from_prediction(self, prediction, cluster_indices):
        # Prediction:
        # [
        #    {
        #        'cluster_count': [0.2, 0.3, 0.5],
        #        'elements': [
        #            {
        #                k_min: [0.3, 0.7],
        #                ...
        #                k_max: [0.3, 0.3, 0.3, 0.1]
        #            },
        #            ...
        #        ]
        #    },
        #    ...
        # ]
        # Cluster indices:
        # [[0, 1, 0, ... 2], [1, 2, 0, ...]...]
        # = For the nth element the cluster index

        # Handle list inputs
        if isinstance(prediction, list):
            return list(map(
                lambda i: self.evaluate_metrics_from_prediction(prediction[i], cluster_indices[i]),
                range(len(prediction))
            ))

        # Create an output structure like:
        # {
        #   'metric_A': {
        #     k_min: 0.2,
        #     ...
        #     k_max: 0.4
        #   },
        #   'metric_B': {
        #     k_min: 0.3,
        #     ...
        #     k_max: 0.5
        #   }
        # }
        metrics = {metric: {} for metric in self._evaluation_metrics.keys()}
        cluster_counts = self._get_cluster_counts()  # list(self._data_provider.get_cluster_counts())
        for cluster_count in cluster_counts:

            # Build the cluster indices structure for the prediction for "cluster_count" clusters
            p_cluster_indices = list(map(
                lambda p: np.argmax(p[cluster_count]),
                prediction['elements']
            ))

            # Calculate all metrics
            for metric, f_metric in self._evaluation_metrics.items():
                metrics[metric][cluster_count] = f_metric(cluster_indices, p_cluster_indices)

        return metrics

    def __hints_to_np_arr(self, hints, count=1):
        if hints is None:
            hints = [None] * count
        else:
            count = len(hints)
        X_hints = [self.__preprocess_hints(c_hints) for c_hints in hints]
        X_hints = np.asarray(X_hints)
        return X_hints

    def __preprocess_hints(self, hints):
        """
        This function preprocesses the hints. It creates an upper similarity matrix out of the hints.
        The diagonal is not included (it obviously would be everywhere 1).

        The matrix is flattened to an array (assuming there are 4 inputs):
        [
            (x0 is in the same cluster as x1),
            (x0 is in the same cluster as x2),
            (x0 is in the same cluster as x3),
            (x1 is in the same cluster as x2),
            (x1 is in the same cluster as x3),
            (x2 is in the same cluster as x3),
        ]

        If both elements are in the same cluster, the value is 1, otherwise it is 0.
        :param batch_size:
        :param hints:
        :return:
        """
        n = self.input_count
        result = np.zeros((n * (n - 1) // 2,), dtype=np.float32)
        if hints is None:
            return result

        # Check if some elements occur more than once in the hints
        dbl_elements = [item for item, count in collections.Counter(list(chain(*hints))).items() if count > 1]
        if len(dbl_elements) > 0:
            raise Exception("Some elements are more than once in the hints: {}; Hints: {}".format(dbl_elements, hints))

        # Add all hints (it could be done more efficient by just setting the right index, but it would be more complicated to develop (and to read (and speed doesnt matter here at all)))
        processes_hints = {source_i:set() for source_i in range(self.input_count)}
        def add_hint(x, y):
            min_i = min(x, y)
            max_i = max(x, y)
            processes_hints[min_i].add(max_i)
        for hint in hints:
            for x, y in combinations(hint, r=2):
                add_hint(x, y)
        i = 0
        for source_i in range(self.input_count):
            curr_processed_hints = processes_hints[source_i]
            for target_i in range(source_i + 1, self.input_count):
                if target_i in curr_processed_hints:
                    result[i] = 1.
                i += 1
        return result


    def predict(self, X, hints=None, debug_mode=None, debug_outputs=None):
        # Cases:
        # X is a list of lists or np-arrays -> multiple runs
        # X is a list of np-arrays -> single run
        # Assuming for now: it is a list of lists of arrays
        # [
        #   [run1input0, run1input1, ...],
        #   [run2input0, run2input1, ...],
        # ]
        #
        # Important:
        # If X contains less inputs than the neural networks expects (e.g. len(X[0]) < self._input_count), then some
        # inputs will be 0.
        #
        # What does the input "hints" stand for?
        # It contains hints for the network which elements are in the same cluster. It is a list of lists of lists:
        # [[[[1, 3, 4], [0, 2]]], ...]
        # The most outer list is just for the batch hints[0] contains hints for the first input, hint[2] for the second,
        # etc. The value "None" is allowed for the hints variable, and also for the elements inside the hints-variable.
        # We focus now on the hints for X[0], this means hints[0]:
        # This example indicates that for sure (=the hint) the elements 1, 3 and 4 are in the same cluster. The second
        # hint is that 0 and 2 are also in the same cluster. If an element index does not show up in this list, this means
        # it is absolutely nothing known about the position of the input. The network may use these hints, but they also
        # could be ignored (this highly depends on how the network is implemented)
        if debug_mode is None:
            debug_mode = self.debug_mode

        X_hints = self.__hints_to_np_arr(hints, len(X))

        data_shape = self.data_provider.get_data_shape()
        X_preprocessed = [
            np.zeros((len(X),) + data_shape, dtype=np.float32) for i in range(self._input_count)
        ]
        for c in range(len(X)):
            ci = X[c]
            if len(ci) != self._input_count:
                print("len(ci)={}, but self._input_count={}: Unwanted behaviour may occur!".format(len(ci), self._input_count))
            for i in range(min(len(ci), self._input_count)):
                X_preprocessed[i][c] = self._normalize_array_if_required(X[c][i])

        # Add the hints input
        # TODO (this breaks the current interface!); Maybe add a method get_clustering_hints() to get the hints inside
        # the build neural network function
        nw_input = X_preprocessed + [X_hints]

        # obsolete: TODO: Prepare X (use it directly from the data provider)
        prediction = self._model_prediction.predict(nw_input, batch_size=self._minibatch_size)

        prediction_debug_output_count = len(self._prediction_debug_outputs)
        additional_prediction_output_count = len(self._additional_prediction_outputs)

        # Split the output: "Normal output", debug and additional outputs
        if (prediction_debug_output_count + additional_prediction_output_count) > 0:
            prediction_outputs = prediction[:-(prediction_debug_output_count + additional_prediction_output_count)]
        else:
            prediction_outputs = prediction
        if additional_prediction_output_count > 0:
            curr_debug_outputs = prediction[-(prediction_debug_output_count + additional_prediction_output_count):-additional_prediction_output_count]
            additional_prediction_outputs = prediction[-additional_prediction_output_count:]
        else:
            curr_debug_outputs = prediction[-(prediction_debug_output_count + additional_prediction_output_count):]
            additional_prediction_outputs = []

        # Check debug outputs.
        if prediction_debug_output_count > 0:

            # Extract and remove the debug outputs
            # debug_outputs = prediction[-prediction_debug_output_count:]
            # prediction = prediction[:-prediction_debug_output_count]

            # If the debug mode is enabled: Print the debug values
            if debug_mode:
                print("~~~~~~~~~~~~~~~~~~~~~")
                print("~Debug output: Start~")
                print("~~~~~~~~~~~~~~~~~~~~~")
                print("Minibatch size: {}".format(self.minibatch_size))
                print("Input count: {}".format(self.input_count))
                print()
                with np_show_complete_array():
                    for i in range(prediction_debug_output_count):
                        debug_output = self._prediction_debug_outputs[i]
                        if i > 0:
                            print()
                        if 'name' in debug_output and debug_output['name'] is not None:
                            print("Name: {}".format(debug_output['name']))
                        print("Caller: {}:{}".format(debug_output['filename'], debug_output['line']))
                        print("Layer name: {}".format(debug_output['layer'].name))
                        print("Data shape: {}".format(debug_output['layer'].shape))
                        print("Value(s):")
                        print(curr_debug_outputs[i])
                        if self.additional_debug_array_printer is not None:
                            print("Additional debug array printer output:")
                            print(self.additional_debug_array_printer(curr_debug_outputs[i]))
                        if debug_outputs is not None:
                            debug_outputs.append({
                                'layer': self._prediction_debug_outputs[i],
                                'output': curr_debug_outputs[i]
                            })
                print("~~~~~~~~~~~~~~~~~~~~~")
                print("~Debug output: End  ~")
                print("~~~~~~~~~~~~~~~~~~~~~")
                print()


        # Create a structure like this:
        # [
        #    {
        #        'cluster_count': [0.2, 0.3, 0.5],
        #        'elements': [
        #            {
        #                k_min: [0.3, 0.7],
        #                ...
        #                k_max: [0.3, 0.3, 0.3, 0.1]
        #            },
        #            ...
        #        ]
        #    },
        #    ...
        # ]

        cluster_counts = self._get_cluster_counts()
        k_min = cluster_counts[0]

        result = []

        # Iterate through each data record of the minibatch
        for r in range(X_preprocessed[0].shape[0]):

            # Get the cluster distribution
            cluster_count = prediction_outputs[-1][r] # The latest element contains the cluster distribution
            elements = []

            # Go through each input element
            for i in range(self._input_count):
                clusters = {}

                # Go through each possible cluster count
                for k in cluster_counts:
                    # Index:
                    # len(cluster_count) * i +  # For each element we get len(cluster_count) distributions
                    # (k - k_min)               # The index for the cluster k
                    distribution = prediction_outputs[len(cluster_counts) * i + (k - k_min)][r, 0]
                    clusters[k] = distribution

                elements.append(clusters)

            # Get all additional prediction outputs
            current_additional_prediction_outputs = {}
            for j in range(len(self._additional_prediction_outputs)):
                additional_prediction_output = self._additional_prediction_outputs[j]
                current_additional_prediction_outputs[additional_prediction_output['name']] = additional_prediction_outputs[j][r]

            result.append({
                'cluster_count': cluster_count,
                'elements': elements,
                'additional_outputs': current_additional_prediction_outputs
            })

        return result

    def __reset_debug_outputs(self):
        self._prediction_debug_outputs = []

    def __reset_additional_prediction_outputs(self):
        self._additional_prediction_outputs = []

    def _add_debug_output(self, layer, name=None, create_wrapper_layer=False):
        if create_wrapper_layer:
            layer = Activation('linear')(layer)
        caller = get_caller()
        self._prediction_debug_outputs.append({
            'name': name,
            'layer': layer,
            'filename': caller.filename,
            'line': caller.lineno
        })

    def _add_additional_prediction_output(self, layer, name, create_wrapper_layer=False):
        if create_wrapper_layer:
            layer = Activation('linear')(layer)
        self._additional_prediction_outputs.append({
            'name': name,
            'layer': layer
        })

    def _get_clustering_hint(self):
        return self.__nw_clustering_hint

    def _try_get_additional_build_config_value(self, key, default_value=None):
        if key in self._additional_build_config:
            return self._additional_build_config[key]
        return default_value

    def build_networks(self, print_summaries=False, build_training_model=True, additional_build_config=None):

        if additional_build_config is None:
            additional_build_config = {}
        self._additional_build_config = additional_build_config

        if self._embedding_nn is not None:
            self._embedding_nn.build(self.data_provider.get_data_shape())
            if print_summaries:
                self._embedding_nn.model.summary()

        # Build the prediction model
        nw_input = self.__build_elements_inputs()
        self.__nw_clustering_hint = Input(((self.input_count * (self.input_count - 1)) // 2,))
        nw_output = []
        additional_network_outputs = {}
        self.__reset_debug_outputs()
        self.__reset_additional_prediction_outputs()
        self._build_network(nw_input, nw_output, additional_network_outputs)
        self._model_prediction = Model(
            # nw_input, nw_output +
            nw_input + [self.__nw_clustering_hint], nw_output +
            list(map(lambda a: a['layer'], self._prediction_debug_outputs)) +
            list(map(lambda a: a['layer'], self._additional_prediction_outputs))
        )
        if print_summaries:
            self._model_prediction.summary()
        # self._model_prediction.summary()

        if build_training_model:

            # Build the training model (it is based on the prediction model).
            # This can be quite expensive and is not required if only predictions
            # are done.
            loss_output = []
            self._build_loss_network(nw_output, loss_output, additional_network_outputs)
            self._model_training = Model(nw_input + [self.__nw_clustering_hint], loss_output)
            # self._model_training = Model(nw_input, loss_output)
            if print_summaries:
                self._model_training.summary()
            # self._model_training.summary()

            # Compile the training model
            self._model_training.compile(
                optimizer=self._optimizer,
                loss=self._get_weighted_keras_loss(),
                metrics=self._get_keras_metrics()
            )

            # Register the training model: The prediction model contains only weights which are also available in the
            # training model, therefore we only have to store one of these models. They also share the weights (shared
            # layers) and the weights have therefore only to be loaded for the "larger" model: The training model.
            self._register_model(self._model_training, self._get_name('cluster_nn'))
        else:

            # Register the training model: The training model may contain more weights, but it is not built. Therefore,
            # we register the prediction model. This allows to load its weights from a trained model.
            self._register_model(self._model_prediction, self._get_name('cluster_nn'))

        # Register the embedding model
        if self._embedding_nn is not None:
            self._register_model(self._embedding_nn, self._get_name('embedding_nn'))

        # Register all plots
        self._register_plots()

        # The networks are now built:)
        self._network_built = True

    def test_network(self, count=1, output_directory=None, data_type='test', create_date_dir=True, include_metrics=True, shuffle_data=True):

        # Generate test data
        test_data, test_data_obj_info, test_hints = self._get_data(data_type=data_type, cluster_collection_count=count)
        test_data_X, test_data_hints, test_data_idx = self._data_provider.convert_data_to_prediction_X(test_data, shuffle=shuffle_data)

        # Shuffle the test_data_obj_infos according to the shuffeling
        test_data_obj_info = [(None if x is None else list(chain(*x))) for x in test_data_obj_info]
        test_data_obj_info = [(None if test_data_obj_info[i] is None else [test_data_obj_info[i][j] for j in test_data_idx[i]]) for i in range(len(test_data_obj_info))]

        # Do a prediction
        print("Do a test prediction (output directory: {})...".format(output_directory))
        prediction = self.predict(test_data_X, test_data_hints)

        # Evaluate the metrics if required
        if include_metrics:
            cluster_indices = self.data_to_cluster_indices(test_data, test_data_idx)
            metrics = self.evaluate_metrics_from_prediction(prediction, cluster_indices)
        else:
            metrics = None

        # Summarize
        print("Summarize test results...")
        self._data_provider.summarize_results(test_data_X, test_data, output_directory, prediction, create_date_dir, metrics, additional_obj_info=test_data_obj_info)
        print("Tests done...")

    def dummy_train(self):
        """
        Test the train function with dummy data (only 0).
        :return:
        """
        print("Start dummy training...")
        self.__train_iteration(dummy_train=True)
        print("Finished dummy training...")

    def dummy_predict(self):
        """
        Test the prediction function with dummy data.
        :return:
        """
        data = self.data_provider.get_data(self._input_count, self._minibatch_size, data_type='train', dummy_data=True)
        X, hints, _ = self.data_provider.convert_data_to_prediction_X(data)
        print("Start dummy prediction...")
        self.predict(X, hints)
        print("Finished dummy prediction...")

    def register_autosave(self, output_directory, base_filename=None, nth_iteration=100, always_save_best_config=True,
                          create_examples=True, example_count=4, overwrite_examples=True, include_history=True,
                          print_loss_plot_every_nth_itr=10, train_examples_nth_iteration=None, save_weights_on_early_stop=True,
                          force_saveing_weights_on_early_stop_after_zero_iters=False):
        if base_filename is None:
            base_filename = self._get_name('autosave')

        # Create a function that saves everything thats required
        def f_autosave(suffix, save_weights=True, create_plots=True, test_network_data_type='test'):
            try_makedirs(output_directory)
            base_path = path.join(output_directory, base_filename + '_' + suffix)
            if save_weights:
                self.save_weights(base_path, include_history)
            if create_plots:
                self.save_plots(base_path + '_plot')
            if create_examples:
                example_path = path.join(output_directory, 'examples_' + suffix)
                self.test_network(example_count, example_path, create_date_dir=not overwrite_examples, data_type=test_network_data_type)

        # Should the best configuration always be saved?
        if always_save_best_config:
            self.event_new_best_validation_loss.add(lambda history, loss: f_autosave('best'))

        # Should the configuration anyway be saved from time to time?
        if nth_iteration is not None:
            self.event_training_iteration_after.add(lambda history: f_autosave('itr'), nth=nth_iteration)

        # If an early stop is executed: Should the weights be stored (with the iteration suffix)?
        if save_weights_on_early_stop:
            self.event_early_stopped.add(lambda iterations_done: f_autosave('itr') if (iterations_done > 0) or force_saveing_weights_on_early_stop_after_zero_iters else None)

        # Should some examples with the train data be created from time to time?
        if train_examples_nth_iteration is not None:
            self.event_training_iteration_after.add(
                lambda history: f_autosave('itr_train', save_weights=False, create_plots=True, test_network_data_type='train'),
                nth=train_examples_nth_iteration
            )

        # If defined: How often should the loss plot be created?
        if print_loss_plot_every_nth_itr is not None:
            try_makedirs(output_directory)
            self.event_training_iteration_after.add(lambda history: self.save_plots(
                path.join(output_directory, 'plot')
            ), nth=print_loss_plot_every_nth_itr)

    def try_load_from_autosave(self, output_directory, base_filename=None, config='itr', include_history=True):
        """
        :param output_directory:
        :param base_filename:
        :param config: 'itr' or 'best'
        :return:
        """
        if base_filename is None:
            base_filename = self._get_name('autosave')
        if config == 'itr':
            base_filename += '_itr'
        elif config == 'best':
            base_filename += '_best'
        else:
            raise Exception('Unknown config: {}'.format(config))

        base_path = path.join(output_directory, base_filename)
        # try:
        self.load_weights(base_path, include_history)
        #     return True
        # except Exception:
        #     return False

