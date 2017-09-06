from os import path

import matplotlib.pyplot as plt

import numpy as np

from core.data.data_provider import DataProvider
from impl.data.misc.data_gen_2d import DataGen2dv02

# This class generates simple 2d clusters: random generated centers with gaussian distributed data

class Simple2DPointDataProvider(DataProvider):
    def __init__(self, min_cluster_count=2, max_cluster_count=10, allow_less_clusters=False):
        super().__init__()
        self._dg = DataGen2dv02()
        self._min_cluster_count = min_cluster_count
        self._max_cluster_count = max_cluster_count
        self._allow_less_clusters = allow_less_clusters

    def get_min_cluster_count(self):
        return self._min_cluster_count

    def get_max_cluster_count(self):
        return self._max_cluster_count

    def get_data_shape(self):
        return (2,)

    def get_clusters(self, element_count, cluster_count=None, data_type='train'):
        if cluster_count is not None and cluster_count > self.get_max_cluster_count():
            cluster_count = self.get_max_cluster_count()
        clusters = self._dg.generate(cluster_count=cluster_count, records=element_count,
                                     cluster_count_min=self._min_cluster_count,
                                     cluster_count_max=self._max_cluster_count,
                                     allow_less_clusters=self._allow_less_clusters
                                     )
        return clusters

    def _summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None):
        cluster_counts = self.get_cluster_counts()

        def get_filename(name):
            global fi
            filename = '{:0>2}_{}'.format(get_filename.counter, name)
            get_filename.counter += 1
            return filename
        get_filename.counter = 0

        # What output are generated?
        # - The ground truth
        #   - An image of "clusters"
        #   - A csv file for this with these columns: input_index;cluster_index;x;y
        # - If possible the results:
        #   - Most probable: "The result" (as image)
        #   - All resulting images for all possible cluster counts
        #   - An image of the cluster count distribution
        #   - An csv file with these columns: input_index;real_cluster;predicted_cluster;predicted_cluster_probability;cluster_count=k_min_probabilities;...
        #   - A csv file with these columns (and only one row): real_cluster_count;cluster_count=k_min_probability;cluster_count=k_min_empty_cluster;cluster_count=k_min+1_probability;..
        #
        # This is much to do;) so let us start doing it;)
        # This function could return a hash like:
        # {
        #     'predicted_cluster_count': 2|None,
        #     'predicted_cluster_count_probability': 0.222|None,
        #     'correct_cluster_count': 3
        # }

        # Generate an image of the ground truth
        self.__plot_cluster_image(clusters, path.join(output_directory, get_filename('solution.png')), 'Solution')

        # Generate a csv file for the inputs
        with open(path.join(output_directory, get_filename('input.csv')), 'wt') as f:
            f.write('input_index;cluster_index;x;y\n')
            for i in range(len(X)):
                point = X[i]

                # Unfortunately it is not that easy to get the real cluster, therefore we have to search the points
                # in the clusters array
                # ci = list(
                #     map(lambda cluster: any(map(lambda p: p == point, cluster)), clusters)
                # ).index(True)
                ci = list(
                    map(lambda cluster: sum(map(lambda p: np.array_equal(p, point), cluster)) > 0, clusters)
                ).index(True)

                f.write('{};{};{};{}\n'.format(
                    i, ci, point[0], point[1]
                ))
            f.close()

        if prediction is not None:
            predicted_clusters = self.convert_prediction_to_clusters(X, prediction)
            cluster_probabilities = prediction['cluster_count']

            # Generate an image for the result
            most_probable_cluster_count = np.argmax(cluster_probabilities) + cluster_counts[0]
            self.__plot_cluster_image(
                predicted_clusters[most_probable_cluster_count], # - cluster_counts[0]],
                path.join(output_directory, get_filename('prediction.png')),
                'Prediction'
            )

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

                # Generate the image
                self.__plot_cluster_image(
                    clusters, path.join(output_directory, get_filename(filename + '.png')),
                    additional_title='p={:0.6}'.format(cluster_probabilities[cluster_count - cluster_counts[0]])
                )

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

    def __plot_cluster_image(self, clusters, output_file, additional_title=None):
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
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        empty_clusters = len(list(filter(lambda c: len(c) == 0, clusters)))
        if additional_title is None:
            additional_title = ''
        else:
            additional_title = ': {}'.format(additional_title)
        plt.title('Cluster count: {} (empty clusters: {}){}'.format(len(clusters), empty_clusters, additional_title))
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

if __name__ == '__main__':
    dp = Simple2DPointDataProvider()
    data = dp.get_data(50, 1)
    print(data)

    fig, ax = plt.subplots()
    for cluster in data[0]:
        px = np.asarray(list(map(lambda c: c[0], cluster)))
        py = np.asarray(list(map(lambda c: c[1], cluster)))
        ax.scatter(px, py)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    plt.show(block=True)
