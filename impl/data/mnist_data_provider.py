from os import path
from keras.datasets import mnist
import random

import matplotlib.pyplot as plt

import numpy as np

from core.data.data_provider import DataProvider


class MNISTDataProvider(DataProvider):
    def __init__(self, train_classes=[0, 2, 3, 4, 6, 7], validate_classes=[1, 5, 8, 9], test_classes=[1, 5, 8, 9],
                 min_cluster_count=None, max_cluster_count=None):
        super().__init__()

        self.__data_classes = {
            'train': train_classes,
            'valid': validate_classes,
            'test': test_classes
        }

        # The maximum cluster count is the minimum number of available elements in our data sets (train, test, valid)
        self.__max_cluster_count = min(map(len, self.__data_classes.values()))

        # If the user defined an own max cluster count, try to use it
        if max_cluster_count is not None:
            self.__max_cluster_count = min([self.__max_cluster_count, max_cluster_count])

        # Define the minimum cluster count
        self.__min_cluster_count = 1
        if min_cluster_count is not None:
            self.__min_cluster_count = max([self.__min_cluster_count, min([min_cluster_count, self.__max_cluster_count])])

        # Load the data
        self.__data = None
        self.__load_data()

    def get_min_cluster_count(self):
        return 1

    def get_max_cluster_count(self):
        return self.__max_cluster_count

    def get_data_shape(self):
        return (28, 28, 1)

    def __load_data(self):

        # Load all records
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Merge them (we split them by classes)
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

        # Rehsape x for tensorflow
        x = x.reshape((x.shape[0],) + self.get_data_shape())

        # Normalize x to [0, 1]
        x = x.astype(np.float32) / 255

        # Split the records by classes and store it
        self.__data = {i: x[y == i] for i in range(10)}

    def __get_random_element(self, class_name):
        data = self.__data[class_name]
        return np.reshape(data[random.randint(0, data.shape[0] - 1)], (1,) + data.shape[1:])

    def get_clusters(self, element_count, cluster_count=None, data_type='train'):
        if cluster_count is not None and cluster_count > self.get_max_cluster_count():
            cluster_count = self.get_max_cluster_count()
        if cluster_count is None:
            cluster_count = random.randint(self.__min_cluster_count, self.__max_cluster_count)

        # Choose the correct available classes
        classes = self.__data_classes[data_type]

        # Choose "cluster_count" classes
        classes = np.random.choice(classes, cluster_count, replace=False)

        # Create the clusters and already add one element to each cluster (because every cluster must be non-empty)
        clusters = {class_name: [self.__get_random_element(class_name)] for class_name in classes}
        element_count -= cluster_count

        # Fill now all elements to the data structure
        for i in range(element_count):
            class_name = random.choice(classes)
            clusters[class_name].append(self.__get_random_element(class_name))

        # We need an array of arrays and the class names are no longer relevant
        clusters = list(clusters.values())

        return clusters


    def _summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None):
        cluster_counts = list(self.get_cluster_counts())

        def get_filename(name):
            global fi
            filename = '{:0>2}_{}'.format(get_filename.counter, name)
            get_filename.counter += 1
            return filename

        get_filename.counter = 0

        # # Generate an image of the ground truth
        # self.__plot_cluster_image(clusters, path.join(output_directory, get_filename('solution.png')), 'Solution')

        # Generate a csv file for the inputs (and also a cluster index array; this might be needed later)
        ci_lst = []
        with open(path.join(output_directory, get_filename('input.csv')), 'wt') as f:
            f.write('input_index;cluster_index\n')
            for i in range(len(X)):
                data = X[i]

                # Unfortunately it is not that easy to get the real cluster, therefore we have to search the points
                # in the clusters array
                # ci = list(
                #     map(lambda cluster: any(map(lambda p: p == point, cluster)), clusters)
                # ).index(True)
                ci = list(
                    map(lambda cluster: sum(map(lambda p: np.array_equal(p, data), cluster)) > 0, clusters)
                ).index(True)
                ci_lst.append(ci)

                f.write('{};{}\n'.format(
                    i, ci
                ))
            f.close()

        if prediction is not None:
            predicted_clusters = self.convert_prediction_to_clusters(X, prediction)
            cluster_probabilities = prediction['cluster_count']

            # Generate an image for the result
            most_probable_cluster_count = np.argmax(cluster_probabilities) + cluster_counts[0]

            # TODO: Implement this for images
            # self.__plot_cluster_image(
            #     predicted_clusters[most_probable_cluster_count],  # - cluster_counts[0]],
            #     path.join(output_directory, get_filename('prediction.png')),
            #     'Prediction'
            # )

            def get_point_infos(input_index, cluster_count):
                # Return (input_index, x, y, cluster_index, [cluster0_probability, cluster1_probability, ...])

                # Extract x and y
                p = X[input_index]
                x = p[0]
                y = p[1]

                # Get the probabilities for the clusters
                c_probabilities = prediction['elements'][input_index][cluster_count]

                # Get the cluster index
                cluster_index = np.argmax(c_probabilities)

                # Get all cluster probabilities as array
                c_probabilities = list(c_probabilities)

                # Return everything
                return (
                    input_index, x, y, cluster_index, c_probabilities
                )

            # Generate the cluster distribution image
            self.__plot_cluster_distribution(
                {c: cluster_probabilities[c - cluster_counts[0]] for c in cluster_counts},
                path.join(output_directory, get_filename('cluster_probabilities.png'))
            )

            # Generate the cluster distribution csv file
            with open(path.join(output_directory, get_filename('cluster_probabilities.csv')), 'wt') as f:
                f.write('real_cluster_count;predicted_cluster_count;{}\n'.format(
                    ';'.join(map(lambda c: 'cluster_count={}_probability'.format(c), cluster_counts))
                ))
                f.write('{};{};{}\n'.format(
                    len(clusters), most_probable_cluster_count,
                    ';'.join(map(str, list(cluster_probabilities)))
                ))
                f.close()

            # Generate another cluster distribution file that is nicer to process and that includes all metrics (if there are any)
            with open(path.join(output_directory, get_filename('cluster_probabilities2.csv')), 'wt') as f:
                f.write('cluster_count;probability{}\n'.format(
                    '' if metrics is None else (';' + ';'.join(map(lambda m: 'metric_' + m, sorted(metrics.keys()))))
                ))
                for cluster_count in cluster_counts:
                    f.write('{};{}'.format(
                        cluster_count,
                        cluster_probabilities[cluster_count - cluster_counts[0]]
                    ))
                    if metrics is not None:
                        for metric in sorted(metrics.keys()):
                            f.write(';{}'.format(metrics[metric][cluster_count]))
                    f.write('\n')
                f.close()

            # Generate an image and a csv file for each cluster possibility
            for cluster_count in sorted(list(predicted_clusters.keys())):
                clusters = predicted_clusters[cluster_count]
                filename = 'prediction_{:0>4}'.format(len(clusters))

                # TODO: Implement this for images
                # # Generate the image
                # self.__plot_cluster_image(
                #     clusters, path.join(output_directory, get_filename(filename + '.png')),
                #     additional_title='p={:0.6}'.format(cluster_probabilities[cluster_count - cluster_counts[0]])
                # )

                # Generate the csv file
                with open(path.join(output_directory, get_filename(filename + '.csv')), 'wt') as f:
                    f.write('input_index;x;y;cluster_index;cluster_probability;{}\n'.format(
                        ';'.join(map(lambda c: 'cluster{}_probability'.format(c), range(cluster_count))))
                    )
                    for input_index in range(len(X)):
                        input_index, x, y, cluster_index, c_probabilities = get_point_infos(input_index, cluster_count)
                        f.write('{};{};{};{};{};{}\n'.format(
                            input_index, x, y, cluster_index, c_probabilities[cluster_index],
                            ';'.join(map(str, c_probabilities))
                        ))
                    f.close()

            # Generate additional plots
            if 'additional_outputs' in prediction:  # backward compatibility
                a_i = 0
                for additional_output_name in sorted(prediction['additional_outputs'].keys()):
                    additional_output = prediction['additional_outputs'][additional_output_name]

                    # Check for 2d data: if it is 2d, plot it, otherwise ignore it
                    # If the data is plotted: use the name of the output for the
                    # title of the plot
                    is_2d_data = len(additional_output.shape) == 2 and additional_output.shape[-1] == 2
                    if not is_2d_data:
                        continue

                    xlim = (-1.2, 1.2)
                    ylim = (-1.2, 1.2)
                    if additional_output.shape[0] == len(X):

                        # Create clusters: If the count of points is equal to the input count, assume the points are
                        # transformed inputs. Draw them once in the expected cluster color and once in the predicted
                        # cluster color
                        expected_clusters = [[] for i in range(len(clusters))]
                        predicted_clusters = [[] for i in range(most_probable_cluster_count)]

                        x = []
                        y = []

                        for p_i in range(additional_output.shape[0]):
                            point = additional_output[p_i]
                            expected_clusters[ci_lst[p_i]].append(point)
                            x.append(point[0])
                            y.append(point[1])
                            # predicted_clusters[np.argmax(prediction['elements'][p_i][most_probable_cluster_count]) + cluster_counts[0] - 1].append(point)
                            predicted_clusters[np.argmax(prediction['elements'][p_i][most_probable_cluster_count])].append(
                                point)

                        if min(x) < xlim[0] or max(x) > xlim[1]:
                            xlim = None
                        if min(y) < ylim[0] or max(y) > ylim[1]:
                            ylim = None

                        self.__plot_cluster_image(
                            expected_clusters,
                            path.join(output_directory, get_filename('additional_output_{}_expected'.format(a_i))),
                            additional_title='Additional output "{}": Expected clusters'.format(additional_output_name),
                            use_auto_generated_title=False, xlim=xlim, ylim=ylim
                        )
                        self.__plot_cluster_image(
                            predicted_clusters,
                            path.join(output_directory, get_filename('additional_output_{}_predicted'.format(a_i))),
                            additional_title='Additional output "{}": Predicted clusters'.format(additional_output_name),
                            use_auto_generated_title=False, xlim=xlim, ylim=ylim
                        )

                    else:

                        # Assume the inputs are just "some points". Draw them all in the same color
                        points = []
                        for p_i in range(additional_output.shape[0]):
                            points.append(additional_output[p_i])
                        self.__plot_cluster_image(
                            [points], path.join(output_directory, get_filename('additional_output_{}'.format(a_i))),
                            additional_title='Additional output {}'.format(additional_output_name),
                            use_auto_generated_title=False, xlim=xlim, ylim=ylim
                        )

                    a_i += 1


    def __plot_cluster_image(self, clusters, output_file, additional_title=None, use_auto_generated_title=True,
                             xlim=(-.2, 1.2), ylim=(-.2, 1.2)):
        # Input format:
        # [
        #   [cluster0point0_as_tuple, cluster0point1_as_tuple, ...],
        #   [cluster1point0_as_tuple, ...], ...
        # ]
        fig, ax = plt.subplots()
        for cluster in clusters:
            px = np.asarray(list(map(lambda c: c[0], cluster)))
            py = np.asarray(list(map(lambda c: c[1], cluster)))
            ax.scatter(px, py, alpha=0.8)
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        empty_clusters = len(list(filter(lambda c: len(c) == 0, clusters)))
        if additional_title is None:
            additional_title = ''
        else:
            additional_title = additional_title if not use_auto_generated_title else ': {}'.format(additional_title)
        if use_auto_generated_title:
            auto_title = 'Cluster count: {} (empty clusters: {})'.format(len(clusters), empty_clusters)
        else:
            auto_title = ''
        plt.title('{}{}'.format(auto_title, additional_title))
        plt.savefig(output_file)
        plt.clf()
        plt.close()


    def __plot_cluster_distribution(self, distribution, output_file):
        #
        # Input
        # distribution = {
        #   cluster_count: probability, ...
        # }
        fig, ax = plt.subplots()

        x = list(self.get_cluster_counts())
        y = list(map(lambda xi: distribution[xi], x))

        # print("X: {}".format(x))
        # print("Y: {}".format(y))
        # print("Y_o: {}".format(distribution))

        ax.bar(x, y, 0.9, color="blue")
        plt.show(block=False)
        plt.savefig(output_file)
        # print("{} saved...".format(output_file))

        plt.clf()
        plt.close()
