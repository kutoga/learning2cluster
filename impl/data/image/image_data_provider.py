from os import path
from keras.datasets import mnist
import random

from itertools import chain

from core.helper import try_makedirs
import shutil

import matplotlib.pyplot as plt
from scipy.misc import imsave
from yattag import Doc

import numpy as np

from core.data.data_provider import DataProvider


class ImageDataProvider(DataProvider):
    def __init__(self, train_classes, validate_classes, test_classes,
                 min_cluster_count=None, max_cluster_count=None, auto_load_data=True,
                 return_1d_images=False, center_data=False, random_mirror_images=False):
        super().__init__()
        self.__return_1d_images = return_1d_images
        self._center_data = center_data
        self._random_mirror_images = random_mirror_images

        self._data_classes = {
            'train': train_classes,
            'valid': validate_classes,
            'test': test_classes
        }

        # The maximum cluster count is the minimum number of available elements in our data sets (train, test, valid)
        self._max_cluster_count = min(map(len, self._data_classes.values()))

        # If the user defined an own max cluster count, try to use it
        if max_cluster_count is not None:
            self._max_cluster_count = min([self._max_cluster_count, max_cluster_count])

        # Define the minimum cluster count
        self.__min_cluster_count = 1
        if min_cluster_count is not None:
            self.__min_cluster_count = max([self.__min_cluster_count, min([min_cluster_count, self._max_cluster_count])])

        self.__data = None
        if auto_load_data:
            # Load the data
            self.load_data()

    @property
    def center_data(self):
        return self._center_data

    @center_data.setter
    def center_data(self, center_data):
        self._center_data = center_data

    def _scale_data(self, data, min_value=0, max_value=255):
        data = data.astype(np.float32)
        data -= min_value
        data /= (max_value - min_value)
        if self._center_data:
            data = (data - 0.5) * 2
        return data

    def _unscale_data(self, data, target_min_value=0, target_max_value=255, target_type=np.uint8):
        if self._center_data:
            data = (data / 2) + 0.5
        data *= (target_max_value - target_min_value)
        data += target_min_value
        data = data.astype(target_type)
        return data

    def _get_img_data_shape(self):
        pass

    def get_data_shape(self):
        img_data_shape = self._get_img_data_shape()
        if self.__return_1d_images:
            img_data_shape = img_data_shape[:2]
        return img_data_shape

    def get_min_cluster_count(self):
        return self.__min_cluster_count

    def get_max_cluster_count(self):
        return self._max_cluster_count

    def load_data(self):
        self.__data = self._load_data()

    def _get_data(self):
        return self.__data

    def _load_data(self):
        pass

    def _get_random_element(self, class_name):
        data = self.__data[class_name]
        element = np.reshape(data[random.randint(0, data.shape[0] - 1)], (1,) + data.shape[1:])
        if self._random_mirror_images and bool(random.getrandbits(1)):
            element = np.fliplr(element)
        additional_obj_info = {
            'description': class_name,
            'class': class_name
        }
        return element, additional_obj_info

    def __image_2d_to_1d(self, img):
        if len(img.shape) != 2:
            if img.shape[2] != 1:
                raise ValueError()
            img = np.reshape(img, img.shape[:2])
        return img

    def __image_1d_to_2d(self, img):
        img = np.reshape(img, img.shape + (1,))
        return img

    def _get_clusters(self, element_count, cluster_count=None, data_type='train'):
        if cluster_count is not None and cluster_count > self.get_max_cluster_count():
            cluster_count = self.get_max_cluster_count()
        if cluster_count is None:
            cluster_count = random.randint(self.__min_cluster_count, self._max_cluster_count)

        # Choose the correct available classes
        classes = self._data_classes[data_type]

        # Choose "cluster_count" classes
        classes = np.random.choice(classes, cluster_count, replace=False)

        # Create the clusters and already add one element to each cluster (because every cluster must be non-empty)
        if self.__return_1d_images:
            post_process = lambda element, additional_obj_info: (self.__image_2d_to_1d(element), additional_obj_info)
        else:
            post_process = lambda element, additional_obj_info: (element, additional_obj_info)
        clusters = {class_name: [post_process(*self._get_random_element(class_name))] for class_name in classes}
        element_count -= cluster_count

        # Fill now all elements to the data structure
        for i in range(element_count):
            class_name = random.choice(classes)
            clusters[class_name].append(post_process(*self._get_random_element(class_name)))

        # Create the resulting clusters and the additional_obj_info
        res_clusters = []
        res_additional_obj_info = []
        for k in sorted(clusters.keys()):
            res_clusters.append(list(map(lambda x: x[0], clusters[k])))
            res_additional_obj_info.append(list(map(lambda x: x[1], clusters[k])))

        # # We need an array of arrays. Return it in the order of the classes.
        # res_clusters = [clusters[k] for k in sorted(clusters.keys())]
        #
        # # Return additional object information: The class names
        # additional_obj_info = [[k] * len(clusters[k]) for k in sorted(clusters.keys())]

        return res_clusters, res_additional_obj_info


    def _summarize_single_result(self, X, clusters, output_directory, prediction=None, metrics=None, additional_obj_info=None):
        cluster_counts = list(self.get_target_cluster_counts())
        result = None

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
            result = {}
            if self.__return_1d_images:
                image_post_processor = lambda x: self.__image_1d_to_2d(x)
            else:
                image_post_processor = None
            predicted_clusters, reformatted_additional_obj_infos =\
                self.convert_prediction_to_clusters(X, prediction, point_post_processor=image_post_processor,
                                                    additional_obj_info=additional_obj_info,
                                                    return_reformatted_additional_obj_infos=True)

            cluster_probabilities = prediction['cluster_count']

            def try_get_metrics(cluster_count):
                if metrics is None:
                    return None
                return {k:metrics[k][cluster_count] for k in metrics.keys()}

            # Generate an image for the result
            most_probable_cluster_count = np.argmax(cluster_probabilities) + cluster_counts[0]
            result['most_probable_cluster_count'] = int(most_probable_cluster_count)

            # TODO: Implement this for images
            # self.__plot_cluster_image(
            #     predicted_clusters[most_probable_cluster_count],  # - cluster_counts[0]],
            #     path.join(output_directory, get_filename('prediction.png')),
            #     'Prediction'
            # )
            self.__plot_image_clusters(
                predicted_clusters[most_probable_cluster_count],  # - cluster_counts[0]],
                path.join(output_directory, get_filename('prediction')),
                'Prediction',
                reformatted_additional_obj_infos=(None if reformatted_additional_obj_infos is None else reformatted_additional_obj_infos[most_probable_cluster_count]),
                metrics=try_get_metrics(most_probable_cluster_count)
            )

            def get_point_infos(input_index, cluster_count):
                # Return (input_index, x, y, cluster_index, [cluster0_probability, cluster1_probability, ...])

                # Get the probabilities for the clusters
                c_probabilities = prediction['elements'][input_index][cluster_count]

                # Get the cluster index
                cluster_index = np.argmax(c_probabilities)

                # Get all cluster probabilities as array
                c_probabilities = list(c_probabilities)

                # Return everything
                return (
                    input_index, cluster_index, c_probabilities
                )

            # Generate the cluster distribution image
            cluster_probability_plot_file = path.join(output_directory, get_filename('cluster_probabilities.png'))
            self.__plot_cluster_distribution(
                {c: cluster_probabilities[c - cluster_counts[0]] for c in cluster_counts},
                cluster_probability_plot_file
            )
            result['cluster_probability_plot'] = cluster_probability_plot_file

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
            result['results'] = {}
            for cluster_count in sorted(list(predicted_clusters.keys())):
                current_results = result['results'][int(cluster_count)] = {}
                clusters = predicted_clusters[cluster_count]
                filename = 'prediction_{:0>4}'.format(len(clusters))
                cluster_count_index = cluster_count - cluster_counts[0]
                current_results['probability'] = float(cluster_probabilities[cluster_count_index]) # convert from np.float32 to float (required for json serialization)

                # TODO: Implement this for images
                # # Generate the image
                # self.__plot_cluster_image(
                #     clusters, path.join(output_directory, get_filename(filename + '.png')),
                #     additional_title='p={:0.6}'.format(cluster_probabilities[cluster_count - cluster_counts[0]])
                # )

                # Generate the image

                current_results['file'] = self.__plot_image_clusters(
                    clusters, path.join(output_directory, get_filename(filename)),
                    additional_title='p={:0.6}'.format(cluster_probabilities[cluster_count_index]),
                    reformatted_additional_obj_infos=(None if reformatted_additional_obj_infos is None else reformatted_additional_obj_infos[cluster_count]),
                    metrics=try_get_metrics(cluster_count)
                )

                # Generate the csv file
                with open(path.join(output_directory, get_filename(filename + '.csv')), 'wt') as f:
                    f.write('input_index;cluster_index;cluster_probability;{}\n'.format(
                        ';'.join(map(lambda c: 'cluster{}_probability'.format(c), range(cluster_count))))
                    )
                    for input_index in range(len(X)):
                        input_index, cluster_index, c_probabilities = get_point_infos(input_index, cluster_count)
                        f.write('{};{}\n'.format(
                            input_index,  cluster_index, c_probabilities[cluster_index],
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
        return result


    def _image_plot_preprocessor(self, img):
        return img

    def __plot_image_clusters(self, clusters, output_directory, additional_title=None, use_auto_generated_title=True,
                              reformatted_additional_obj_infos=None, metrics=None):

        # Create the required directories
        shutil.rmtree(output_directory, ignore_errors=True)
        try_makedirs(output_directory)
        img_dir_name = 'img'
        img_dir = path.join(output_directory, img_dir_name)
        try_makedirs(img_dir)
        available_additional_infos = []

        # Store the cluster-images to some subdirectories (this makes it easier to post-process them)
        img_clusters = []
        for i in range(len(clusters)):
            cluster_dir_name = 'cluster{:03d}'.format(i)
            cluster_dir = path.join(img_dir, cluster_dir_name)
            try_makedirs(cluster_dir)
            cluster_objs = clusters[i]
            img_cluster = []
            img_clusters.append(img_cluster)
            for j in range(len(cluster_objs)):
                img = cluster_objs[j]
                img_file_name = 'object{:03d}.png'.format(j)
                img_file = path.join(cluster_dir, img_file_name)

                # Preprocess the image
                img = self._image_plot_preprocessor(img)

                # The image is normalized to [0, 1], denormalize it to [0, 255]
                img = self._unscale_data (img) #img * 255).astype(np.uint8)

                if len(img.shape) > 3:
                    img = img.reshape(img.shape[-3:])
                if img.shape[-1] == 1:
                    img = img.reshape(img.shape[:-1])

                imsave(img_file, img)

                # Save the relative path
                additional_img_info = None
                if reformatted_additional_obj_infos is not None:
                    additional_img_info = reformatted_additional_obj_infos[i][j]
                    available_additional_infos.append(additional_img_info)
                img_cluster.append({
                    'img_path': path.join(img_dir_name, cluster_dir_name, img_file_name),
                    'additional_info': additional_img_info
                })

        # Generate the title
        all_images = list(chain(*img_clusters))
        empty_clusters = len(list(filter(lambda c: len(c) == 0, clusters)))
        if additional_title is None:
            additional_title = ''
        else:
            additional_title = additional_title if not use_auto_generated_title else ': {}'.format(additional_title)
        if use_auto_generated_title:
            auto_title = 'Cluster count: {} (empty clusters: {}, object count: {})'.format(len(clusters), empty_clusters, len(all_images))
        else:
            auto_title = ''
        title = '{}{}'.format(auto_title, additional_title)

        # Create now a "nice looking" html file
        colors = self.__get_html_color_pairs(list(map(lambda x: x['class'], available_additional_infos)))
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag('head'):
                with tag('style', type="text/css"):
                    pass
            with tag('body'):
                with tag('h1'):
                    text(title)

                with tag('table', border="0"):
                    with tag('tr'):
                        if all(map(lambda img: img['additional_info'] is not None, all_images)):
                            with tag('td', style="vertical-align: top;"):
                                with tag('h2'):
                                    text('Correct Clusters')
                                with tag('table', border='1'):
                                    with tag('tr'):
                                        with tag('th'):
                                            text('Cluster Index')
                                        with tag('th'):
                                            text('Object Count')
                                        with tag('th'):
                                            text('Class')
                                    ci = 0
                                    for class_name in sorted(colors.keys()):
                                        with tag('tr'):
                                            with tag('td'):
                                                text(ci)
                                            with tag('td'):
                                                text(len(list(filter(lambda img: img['additional_info']['class'] == class_name, all_images))))
                                            with tag('td', bgcolor=colors[class_name]):
                                                text(str(class_name))
                                        ci += 1

                        if metrics is not None:
                            with tag('td', width="10px"):
                                pass
                            with tag('td', style="vertical-align: top;"):
                                with tag('h2'):
                                    text('Metrics')
                                with tag('table', border='1'):
                                    with tag('tr'):
                                        with tag('th'):
                                            text('Name')
                                        with tag('th'):
                                            text('Value')
                                    for metric in sorted(metrics.keys()):
                                        with tag('tr'):
                                            with tag('td'):
                                                text(metric)
                                            with tag('td'):
                                                text("{:.10f}".format(metrics[metric]))

                with tag('h2'):
                    text('Predicted Clusters')
                with tag('table', border='1'):
                    with tag('tr'):
                        with tag('th'):
                            text('Cluster Index')
                        with tag('th'):
                            text('Object Count')
                        with tag('th'):
                            text('Objects')
                    for i in range(len(img_clusters)):
                        img_cluster = img_clusters[i]

                        # Try to sort the cluster
                        if not any(map(lambda img: img['additional_info'] is None, img_cluster)):
                            img_cluster = sorted(img_cluster, key=lambda img: img['additional_info']['class'])

                        with tag('tr'):
                            with tag('td'):
                                text('{}'.format(i))
                            with tag('td'):
                                text('{}'.format(len(img_cluster)))
                            with tag('td'):
                                with tag('table'):
                                    with tag('tr'):
                                        for img in img_cluster:
                                            bgcolor = '#FFFFFF'
                                            if img['additional_info'] is not None:
                                                bgcolor = colors[img['additional_info']['class']]
                                            with tag('td', bgcolor=bgcolor):
                                                with tag('center'):
                                                    with tag('img', src=img['img_path'], width='95px'):
                                                        pass
                                    with tag('tr'):
                                        for img in img_cluster:
                                            bgcolor = '#FFFFFF'
                                            if img['additional_info'] is not None:
                                                bgcolor = colors[img['additional_info']['class']]
                                            with tag('td', bgcolor=bgcolor, style='width: 100px; overflow: hidden;'):
                                                if img['additional_info'] is not None:
                                                    text(str(img['additional_info']['description']))
        html = doc.getvalue()

        # Save the html file
        output_file = path.join(output_directory, 'index.html')
        with open(output_file, "w") as fh:
            fh.write(html)
        return output_file

    def __get_html_color_pairs(self, keys):
        html_colors = list(self.__get_html_colors())
        random.shuffle(html_colors)
        mapping = {k: '#FFFFFF' for k in keys}
        keys = list(set(keys))
        for i in range(min(len(keys), len(html_colors))):
            mapping[keys[i]] = html_colors[i]
        return mapping

    def __get_html_colors(self):
        return {
            '#00FFFF', '#F5F5DC', '#8A2BE2', '#A52A2A', '#7FFF00',
            '#FF7F50', '#6495ED', '#008B8B', '#B8860B', '#006400',
            '#FF8C00', '#8FBC8F', '#FF1493', '#696969', '#FFD700'
        }

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

        x = list(self.get_target_cluster_counts())
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
