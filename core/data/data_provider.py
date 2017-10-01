from random import Random
from itertools import chain
from os import path
from datetime import datetime
from math import log10, ceil

import numpy as np

from core.helper import try_makedirs

class DataProvider:
    def __init__(self, seed=None):
        self.__rand = Random(seed)

    def get_min_cluster_count(self):
        pass

    def get_max_cluster_count(self):
        pass

    def get_cluster_count_range(self):
        return self.get_min_cluster_count(), self.get_max_cluster_count()

    def get_cluster_counts(self):
        """
        Return all possible cluster counts in an ascending order. The returned object is iterable.
        :return: The possible cluster counts.
        """
        return range(self.get_min_cluster_count(), self.get_max_cluster_count() + 1)

    def get_cluster_counts_distribution(self):
        """
        Get the probabilities for each cluster count. The default implementation uses equally distributed cluster
        counts
        :return:
        """
        cluster_counts = self.get_cluster_counts()
        p = 1. / len(cluster_counts)
        return {
            c: p for c in cluster_counts
        }

    def get_data_shape(self):
        pass

    def convert_data_to_prediction_X(self, data, shuffle=True, return_shuffle_indices=False):
        """
        Convert data that has the format of "get_data" to "X" input data for the prediction. The "data" value
        contains already the perfect result about the clusters and the "X" data only contains the input points and
        no information about any cluster.

        :param data:
        :return:
        """

        # Generate the X data set which can directly be used for predictions (or whatever)
        # X = [
        #     # This list contains all input data sets (one set: N points to cluster)
        #     [
        #         # This list contains all input points
        #     ]
        # ]
        # Do this in a public function (so anyone may use it)

        X = []
        shuffle_indices = []
        for cluster_collection in data:
            inputs = list(chain.from_iterable(cluster_collection))

            shuffle_idx = list(range(len(inputs)))
            if shuffle:
                self.__rand.shuffle(shuffle_idx)
                inputs = [inputs[i] for i in shuffle_idx]
            X.append(inputs)
            shuffle_indices.append(shuffle_idx)

        if return_shuffle_indices:
            return X, shuffle_indices
        else:
            return X

    def convert_prediction_to_clusters(self, X, prediction):

        # Handle list inputs
        if isinstance(prediction, list):
            return list(map(lambda i: self.convert_data_to_prediction_X(X[i], prediction[i]), range(len(prediction))))

        cluster_counts = list(self.get_cluster_counts())
        # clusters = []

        # TODO: The following code was first created as a loop and modified -> cleanup
        current_inputs = X
        current_prediction = prediction
        current_cluster_combinations = {}

        for ci in cluster_counts:
            current_clusters = [[] for i in range(ci)]
            #probability = current_prediction['cluster_count'][cluster_counts[ci - cluster_counts[0]]]
            for ii in range(len(X)):
                point = current_inputs[ii]
                cluster_index = np.argmax(current_prediction['elements'][ii][ci])
                current_clusters[cluster_index].append(point)
            current_cluster_combinations[ci] = current_clusters
        # clusters.append(current_cluster_combinations)

        return current_cluster_combinations

    def get_data(self, elements_per_cluster_collection, cluster_colletion_count,
                       cluster_count_f=None, cluster_count=None, cluster_count_range=None,
                       dummy_data=False, data_type='train'):
        """

        :param elements_per_cluster_collection:
        :param cluster_colletion_count:
        :param cluster_count_f:
        :param cluster_count:
        :param cluster_count_range:
        :param dummy_data:
        :param data_type: 'train', 'valid' or 'test'
        :return:
        """

        # Define the max cluster count function: If it is already given we are done
        if cluster_count_f is None:
            if cluster_count_range is not None:
                # Use a range of cluster counts
                cluster_count_f = lambda: self.__rand.randint(*cluster_count_range)
            elif cluster_count is not None:
                # Use a fixed max cluster count
                cluster_count_f = lambda: cluster_count
            else:
                # Use a default value for the cluster count (=None)
                cluster_count_f = lambda: None

        if dummy_data:

            # Generate some dummy 0-data.
            data_shape = self.get_data_shape()
            train_data = []
            for i in range(cluster_colletion_count):

                # Generate cluster_count_f() clusters, or if no value is defined: Use the maximum cluster count
                cluster_count = cluster_count_f()
                if cluster_count is None:
                    cluster_count = self.get_max_cluster_count()
                clusters = [[] for c in range(cluster_count)]

                # Assign each object to a cluster
                for o in range(elements_per_cluster_collection):
                    clusters[o % cluster_count].append(np.zeros(data_shape, dtype=np.float32))

                # Remove empty clusters
                clusters = list(filter(lambda c: len(c) > 0, clusters))

                train_data.append(clusters)

        else:

            # Generate the training data
            train_data = [
                self.get_clusters(elements_per_cluster_collection, cluster_count_f(), data_type=data_type)
                for i in range(cluster_colletion_count)
            ]

        return train_data

    def get_clusters(self, element_count, cluster_count=None, data_type='train'):
        """
        Generate some clusters and return them. Format [[obj1cluster1, obj2cluster1, ...], [obj1cluster2, ...]]
        :param element_count:
        :param cluster_count:
        :param test_data
        :return:
        """
        pass

    def summarize_results(self, X, clusters, output_directory, prediction=None, create_date_dir=True, metrics=None):
        """
        Summarize results and store the results to a defined output directory.
        :param X:
        :param clusters:
        :param output_directory:
        :param prediction:
        :return:
        """
        input_records = len(clusters)

        # Create the top directory
        try_makedirs(output_directory)

        # Create a directory with the current date / time (if required)
        if create_date_dir:
            output_directory = path.join(output_directory, datetime.now().strftime("%Y%m%d%H%M%S"))
            try_makedirs(output_directory)

        # Analyze all tests
        digits = 1
        if input_records > 0:
            digits = int(ceil(log10(input_records) + 0.1))
        output_directory_name = 'test{:0' + str(digits) + 'd}'
        for i in range(len(clusters)):

            current_X = X[i] #list(map(lambda X_i: X_i[i], X))
            current_clusters = clusters[i]
            current_output_directory = path.join(output_directory, output_directory_name.format(i))
            current_prediction = None
            if prediction is not None:
                current_prediction = prediction[i]
            current_metrics = None
            if metrics is not None:
                current_metrics = metrics[i]

            self.summarize_single_result(
                current_X, current_clusters, current_output_directory, current_prediction, current_metrics
            )

        # Create a summary over everything
        # TBD

    def summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None):

        # Create the output directory
        try_makedirs(output_directory)

        # Call the implementation
        self._summarize_single_result(X, clusters, output_directory, prediction, metrics)

    def _summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None):
        pass



