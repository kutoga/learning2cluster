from itertools import chain
from random import Random
from os import path
from time import time

import numpy as np

from termcolor import colored

from keras.models import Model
from keras.layers import Input

from core.nn.base_nn import BaseNN
from core.nn.history import History
from core.nn.helper import filter_None, AlignedTextTable
from core.event import Event
from core.helper import try_makedirs


class ClusterNN(BaseNN):
    def __init__(self, data_provider, input_count, embedding_nn=None, seed=None):
        BaseNN.__init__(self, name='NN_[CLASS]_I{}'.format(input_count))
        self._rand = Random(seed)
        self._data_provider = data_provider
        self._input_count = input_count
        self._embedding_nn = embedding_nn

        self._optimizer = 'adadelta'
        self._minibatch_size = 100

        self._train_history = History()
        self._validate_every_nth_epoch = 10

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

        # The cluster count function. It may be overwritten (default=random)
        self._f_cluster_count = None

        # Debug outputs
        self._prediction_debug_outputs = None

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

    def _get_embedding(self, layer):

        # If a list of layers is given: Return the embedding for each layer
        if isinstance(layer, list):
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
            plt.plot(
                *filter_None(x, history['loss']),
                *filter_None(x, history['val_loss']),
                alpha=0.7,
                lw=0.5
            )
            plt.legend(['loss: training', 'loss: validation'])
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.grid(True)
        self._register_plot(model_name, loss_plot)

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
        self._register_plot(model_name, best_loss_plot)

    def _build_network(self, network_input, network_output, additional_network_outputs, debug_output):
        return None

    def _build_loss_network(self, network_output, loss_output, additional_network_outputs):
        return None

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
            for i in range(len(current_inputs)):
                X[i][c] = current_inputs[i][0]
        return X

    def __build_Xy_data(self, data):

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

            # That's it for the current cluster collection:)
            inputs.append({
                'cluster_count': len(clusters),
                'data': current_inputs
            })

        return self.__build_X_data(inputs), self._build_y_data(inputs)

    def __build_inputs(self):
        return [
            self._s_layer('input_{}'.format(i), lambda name: Input(shape=self.data_provider.get_data_shape(), name=name))
            for i in range(self._input_count)
        ]

    def __get_last_epoch(self):
        return self._get_history(self._model_training).length()

    def __get_data(self, data_type='train', dummy_data=False, cluster_collection_count=None, *args):
        if cluster_collection_count is None:
            cluster_collection_count = self._minibatch_size
        return self._data_provider.get_data(self._input_count, cluster_collection_count, data_type=data_type, dummy_data=dummy_data, cluster_count_f=self._f_cluster_count, *args)

    def __train_iteration(self, dummy_train=False):
        self.event_training_iteration_before.fire(nth=self.__get_last_epoch())
        do_validation = (self.__get_last_epoch() + 1) % self._validate_every_nth_epoch == 0

        # Generate training data
        t_start_data_gen_time = time()
        train_data = self.__get_data('train', dummy_data=dummy_train) # self._data_provider.get_data(self._input_count, self._minibatch_size, data_type='train', dummy_data=dummy_train, max_cluster_count_f=self._f_cluster_count)
        X_train, y_train = self.__build_Xy_data(train_data)

        # If required: Generate validation data
        if do_validation:
            valid_data = self.__get_data('valid', dummy_data=dummy_train) # self._data_provider.get_data(self._input_count, self._minibatch_size, data_type='valid', dummy_data=dummy_train, max_cluster_count_f=self._f_cluster_count)
            validation_data = self.__build_Xy_data(valid_data)
        else:
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
        train = {}
        valid = {}
        others = {}
        for key in history.keys():
            value = history[key][-1]
            if value is None:
                continue
            if key == "time_s":
                others[key] = value
            elif "val_" in key:
                valid[key] = value
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
                        tbl.add_cell(colored("{:0.6}".format(value), color=color, attrs=attrs))
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
        for i in range(iterations):
            self.__train_iteration()

    def predict(self, X):
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

        data_shape = self.data_provider.get_data_shape()
        X_preprocessed = [
            np.zeros((len(X),) + data_shape, dtype=np.float32) for i in range(self._input_count)
        ]
        for c in range(len(X)):
            ci = X[c]
            if len(ci) != self._input_count:
                print("len(ci)={}, but self._input_count={}: Unwanted behaviour may occur!".format(len(ci), self._input_count))
            for i in range(min(len(ci), self._input_count)):
                X_preprocessed[i][c] = X[c][i]

        # TODO: Prepare X (use it directory from the data provider)
        prediction = self._model_prediction.predict(X_preprocessed, batch_size=self._minibatch_size)

        # Check debug outputs.
        prediction_debug_output_count = len(self._prediction_debug_outputs)
        if prediction_debug_output_count > 0:

            # Extract and remove the debug outputs
            debug_outputs = prediction[-prediction_debug_output_count:]
            prediction = prediction[:-prediction_debug_output_count]

            # If the debug mode is enabled: Print the debug values
            if self.debug_mode:
                print("PRINT DEBUG VALUES, count: {}".format(prediction_debug_output_count))
                pass


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

        cluster_counts = list(self._data_provider.get_cluster_counts())
        k_min = cluster_counts[0]

        result = []

        # Iterate through each data record of the minibatch
        for r in range(X_preprocessed[0].shape[0]):

            # Get the cluster distribution
            cluster_count = prediction[-1][r] # The latest element contains the cluster distribution
            elements = []

            # Go through each input element
            for i in range(self._input_count):
                clusters = {}

                # Go through each possible cluster count
                for k in cluster_counts:
                    # Index:
                    # len(cluster_count) * i +  # For each element we get len(cluster_count) distributions
                    # (k - k_min)               # The index for the cluster k
                    distribution = prediction[len(cluster_counts) * i + (k - k_min)][r, 0]
                    clusters[k] = distribution

                elements.append(clusters)

            result.append({
                'cluster_count': cluster_count,
                'elements': elements
            })

        return result

    def build_networks(self):
        if self._embedding_nn is not None:
            self._embedding_nn.build(self.data_provider.get_data_shape())

        # Build the prediction model
        nw_input = self.__build_inputs()
        nw_output = []
        additional_network_outputs = {}
        prediction_debug_output = []
        self._build_network(nw_input, nw_output, additional_network_outputs, prediction_debug_output)
        self._model_prediction = Model(nw_input, nw_output + prediction_debug_output)
        self._prediction_debug_outputs = prediction_debug_output
        # self._model_prediction.summary()

        # Build the training model (it is based on the prediction model)
        loss_output = []
        self._build_loss_network(nw_output, loss_output, additional_network_outputs)
        self._model_training = Model(nw_input, loss_output)
        # self._model_training.summary()

        # Compile the training model
        self._model_training.compile(
            optimizer=self._optimizer,
            loss=self._get_keras_loss(),
            metrics=self._get_keras_metrics()
        )

        # Register the embedding model
        if self._embedding_nn is not None:
            self._register_model(self._embedding_nn, self._get_name('embedding_nn'))

        # Register the training model: The prediction model contains only weights which are also available in the
        # training model, therefore we only have to store one of these models. They also share the weights (shared
        # layers) and the weights have therefore only to be loaded for the "larger" model: The training model.
        self._register_model(self._model_training, self._get_name('cluster_nn'))

        # Register all plots
        self._register_plots()

        # TODO: Do the discriminator things

        # DUMMY
        # self.dummy_predict()
        # self.dummy_train()

        pass

    def test_network(self, count=1, output_directory=None, data_type='test', create_date_dir=True):

        # Generate test data
        test_data = self.__get_data(data_type=data_type, cluster_collection_count=count) # self._data_provider.get_data(self._input_count, count, data_type=data_type)
        test_data_X = self._data_provider.convert_data_to_X(test_data)

        # Do a prediction
        print("Do a test prediction (output directory: {})...".format(output_directory))
        prediction = self.predict(test_data_X)

        # Summarize
        print("Summarize test results...")
        self._data_provider.summarize_results(test_data_X, test_data, output_directory, prediction, create_date_dir)
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
        X = self.data_provider.convert_data_to_X(data)
        print("Start dummy prediction...")
        self.predict(X)
        print("Finished dummy prediction...")

    def register_autosave(self, output_directory, base_filename=None, nth_iteration=100, always_save_best_config=True,
                          create_examples=True, example_count=4, overwrite_examples=True, include_history=True,
                          print_loss_plot_every_nth_itr=1):
        if base_filename is None:
            base_filename = self._get_name('autosave')

        # Create a function that saves everything thats required
        def f_autosave(suffix):
            try_makedirs(output_directory)
            base_path = path.join(output_directory, base_filename + '_' + suffix)
            self.save_weights(base_path, include_history)
            self.save_plot(base_path + '_history.png')
            if create_examples:
                example_path = path.join(output_directory, 'examples_' + suffix)
                self.test_network(example_count, example_path, create_date_dir=not overwrite_examples)

        # Should the best configuration always be saved?
        if always_save_best_config:
            self.event_new_best_validation_loss.add(lambda history, loss: f_autosave('best'))

        # Should the configuration anyway be saved from time to time?
        if nth_iteration is not None:
            self.event_training_iteration_after.add(lambda history: f_autosave('itr'), nth=nth_iteration)

        # If defined: How often should the loss plot be created?
        if print_loss_plot_every_nth_itr is not None:
            try_makedirs(output_directory)
            self.event_training_iteration_after.add(lambda history: self.save_plot(
                path.join(output_directory, 'loss_plot.png')
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

