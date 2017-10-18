from random import Random
from itertools import chain
from os import path
from datetime import datetime
from math import log10, ceil

from shutil import rmtree

import numpy as np

import json

from yattag import Doc

from core.helper import try_makedirs

class DataProvider:
    def __init__(self, target_min_cluster_count=None, target_max_cluster_count=None, seed=None):
        self.__rand = Random(seed)
        self.target_min_cluster_count = target_min_cluster_count
        self.target_max_cluster_count = target_max_cluster_count

    def get_min_cluster_count(self):
        pass

    def get_target_min_cluster_count(self):
        if self.target_min_cluster_count is None:
            return self.get_min_cluster_count()
        else:
            return self.target_min_cluster_count

    def get_max_cluster_count(self):
        pass

    def get_target_max_cluster_count(self):
        if self.target_max_cluster_count is None:
            return self.get_max_cluster_count()
        else:
            return self.target_max_cluster_count

    def get_cluster_count_range(self):
        return self.get_min_cluster_count(), self.get_max_cluster_count()

    def get_target_cluster_count_range(self):
        return self.get_target_min_cluster_count(), self.get_target_max_cluster_count()

    def get_cluster_counts(self):
        """
        Return all possible cluster counts in an ascending order. The returned object is iterable.
        :return: The possible cluster counts.
        """
        return range(self.get_min_cluster_count(), self.get_max_cluster_count() + 1)

    def get_target_cluster_counts(self):
        """
        Return all possible cluster counts in an ascending order. The returned object is iterable.
        :return: The possible cluster counts.
        """
        return range(self.get_target_min_cluster_count(), self.get_target_max_cluster_count() + 1)

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

    def convert_prediction_to_clusters(self, X, prediction, point_post_processor=None, additional_obj_info=None,
                                       return_reformatted_additional_obj_infos=False):

        # Handle list inputs
        if isinstance(prediction, list):
            return list(map(lambda i: self.convert_data_to_prediction_X(X[i], prediction[i]), range(len(prediction))))

        cluster_counts = list(self.get_target_cluster_counts())

        # TODO: The following code was first created as a loop and modified -> cleanup
        current_inputs = X
        current_prediction = prediction
        current_cluster_combinations = {}
        reformatted_additional_obj_infos = {}

        for ci in cluster_counts:
            current_clusters = [[] for i in range(ci)]
            current_reformatted_additional_obj_infos = [[] for i in range(ci)]

            #probability = current_prediction['cluster_count'][cluster_counts[ci - cluster_counts[0]]]
            for ii in range(len(X)):
                point = current_inputs[ii]
                if point_post_processor is not None:
                    point = point_post_processor(point)

                cluster_index = np.argmax(current_prediction['elements'][ii][ci])
                current_clusters[cluster_index].append(point)
                if additional_obj_info is not None:
                    current_reformatted_additional_obj_infos[cluster_index].append(additional_obj_info[ii])
            current_cluster_combinations[ci] = current_clusters
            reformatted_additional_obj_infos[ci] = current_reformatted_additional_obj_infos
        # clusters.append(current_cluster_combinations)

        if return_reformatted_additional_obj_infos:
            return current_cluster_combinations, reformatted_additional_obj_infos
        else:
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
            additional_obj_info = [None] * cluster_colletion_count

        else:

            # Generate the training data
            train_data = []
            additional_obj_info = []
            for i in range(cluster_colletion_count):
                data, obj_info = self._get_clusters(elements_per_cluster_collection, cluster_count_f(), data_type=data_type)
                train_data.append(data)
                additional_obj_info.append(obj_info)

        return train_data, additional_obj_info

    def _get_clusters(self, element_count, cluster_count=None, data_type='train'):
        """
        Generate some clusters and return them. Format [[obj1cluster1, obj2cluster1, ...], [obj1cluster2, ...]]
        :param element_count:
        :param cluster_count:
        :param test_data
        :return: clusters, additional_obj_info
        """
        pass

    def summarize_results(self, X, clusters, output_directory, prediction=None, create_date_dir=True, metrics=None, additional_obj_info=None):
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
        results = {}
        for i in range(len(clusters)):

            current_X = X[i] #list(map(lambda X_i: X_i[i], X))
            current_clusters = clusters[i]
            current_output_directory_name = output_directory_name.format(i)
            current_output_directory = path.join(output_directory, current_output_directory_name)
            current_prediction = None
            if prediction is not None:
                current_prediction = prediction[i]
            current_metrics = None
            if metrics is not None:
                current_metrics = metrics[i]
            current_additional_obj_info = None
            if additional_obj_info is not None:
                current_additional_obj_info = additional_obj_info[i]

            current_result = self.summarize_single_result(
                current_X, current_clusters, current_output_directory, current_prediction, current_metrics, current_additional_obj_info
            )
            if current_result is not None:

                def fix_path(path):
                    if path.startswith(current_output_directory):
                        path = './{}/'.format(current_output_directory_name) + path[len(current_output_directory):]
                    return path
                current_result['cluster_probability_plot'] = fix_path(current_result['cluster_probability_plot'])
                for cluster_count in current_result['results'].keys():
                    current_result['results'][cluster_count]['file'] = fix_path(current_result['results'][cluster_count]['file'])

                results[current_output_directory_name] = current_result

        # Create a summary over everything
        if len(results) > 0:
            self._write_test_results_html_file(path.join(output_directory, 'index.html'), results)

    def summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None, additional_obj_info=None):

        # Create the output directory; If it already exists, remove it
        if path.exists(output_directory):
            rmtree(output_directory, ignore_errors=True)
        try_makedirs(output_directory)

        # Call the implementation
        return self._summarize_single_result(X, clusters, output_directory, prediction, metrics, additional_obj_info)

    def _summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None, additional_obj_info=None):
        # Return none or a data structure like:
        # {
        #     'cluster_probability_plot': 'input.png',
        #     'most_probable_cluster_count': 1,
        #     'results': {
        #         1: {
        #             'probability': 0.12,
        #             'file': 'xyz.html'
        #         },
        #         ...
        #     }
        # }
        pass

    def _write_test_results_html_file(self, output_file, test_results):
        """
        test_results = {
            test00: {
                'cluster_probability_plot': 'input.png',
                'most_probable_cluster_count': 1,
                'results': {
                    1: {
                        'probability': 0.12,
                        'file': 'xyz.html'
                    },
                    ...
                }
                ...
            }
        }
        :param output_file:
        :param test_results:
        :return:
        """
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag('head'):
                with tag('style', type="text/css"):
                    text("""
button {
    width: 150px;
    height: 50px;
}
.wbutton {
    width: 450px;
}
                    """)
                with tag('script', src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"):
                    pass
                with tag('script'):
                    text('data={};'.format(json.dumps(test_results)))
                    doc._append("""
                    $(function(){
                        var selected_test = null;

                        // Select all test cases
                        var tests = Object.keys(data);
                        tests.sort();

                        function select_result(title, cluster_count) {
                            $('.current_view_title').text(title);
                            $('.current_view_content').attr('src', selected_test['results'][cluster_count]['file']);
                        }

                        function select_test(test) {
                            selected_test = data[test];
                            $('.test_button').css('background-color', '');
                            selected_test['button'].css('background-color', '#BBBBBB');
                            var target = $('.cluster_count_buttons')
                            target.empty();
                            target.append('<h2>Cluster probabilities</h2>');
                            target.append('<img src="' + selected_test['cluster_probability_plot'] + '" width="450px" />');
                            var cluster_counts = Object.keys(selected_test['results'])
                            cluster_counts.sort()
                            var mpcc = selected_test['most_probable_cluster_count'];
                            function add_btn(text, cluster_count) {
                                var btn = $('<button class="wbutton" type="button">' + text + '</button>');
                                btn.click(function(){select_result(text, cluster_count);});
                                target.append(btn);
                                return btn;
                            }
                            mpcc_btn = add_btn("Prediction (cluster count = " + mpcc + ")", mpcc);
                            cluster_counts.map(function(cluster_count) {
                                add_btn('Cluster count = ' + cluster_count + ", p = " + selected_test['results'][cluster_count]['probability'], cluster_count);
                            });
                            mpcc_btn.click();
                        }

                        // Create some buttons
                        tests.map(function(test){
                            var btn = $('<button class="test_button" type="button" id="' + test +'">' + test + '</button>');
                            data[test]['button'] = btn;
                            btn.click(function(){select_test(test);});
                            $('.test_buttons').append(btn);
                        });
                        select_test(tests[0]);
                    });
                    """)
            with tag('body'):
                with tag('h1'):
                    text('Test collection')
                with tag('div', width="100%", style="background-color:#999999"):
                    with tag('table'):
                        with tag('tr', klass='test_buttons'):
                            pass
                with tag('table', width="100%", height="80%"):
                    with tag('tr'):
                        with tag('td', style="vertical-align: top;", width="500px"):
                            with tag('div', klass='cluster_count_buttons'):
                                pass
                        with tag('td', style="vertical-align: top;"):
                            with tag('h2', klass="current_view_title"):
                                pass
                            with tag('iframe', klass="current_view_content", frameborder="0px", width="100%", height="100%"):
                                pass
        html = doc.getvalue()

        # Save the html file
        with open(output_file, "w") as fh:
            fh.write(html)


