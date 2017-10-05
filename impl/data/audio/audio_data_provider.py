
# TODO
# Requirements:
# - Output can be a 2D image
# - Output can be a 1D image
# - Allow different audio data types (spectrogram types; maybe use the AUdioHelper from VT2 as an argument for the audio data provider)
import numpy as np
from random import Random
from os import path
from time import time

from impl.data.image.image_data_provider import ImageDataProvider

from impl.data.misc.audio_helper import AudioHelper

from impl.misc.simple_file_cache import SimpleFileCache

class AudioDataProvider(ImageDataProvider):
    def __init__(self, data_dir=None, audio_helper=None, cache_directory=None, train_classes=None, validate_classes=None,
                 test_classes=None, window_width=100, return_1d_audio_data=False,
                 min_cluster_count=None, max_cluster_count=None, concat_audio_files_of_speaker=False):
        if audio_helper is None:
            audio_helper = AudioHelper()

        self.__rand = Random()
        self.__data = None
        self.__data_dir = data_dir
        self.__audio_helper = audio_helper
        self.__cache = None if cache_directory is None else SimpleFileCache(cache_directory, compression=True)
        self.__window_width = window_width
        self.__concat_audio_files_of_speaker = concat_audio_files_of_speaker

        self._load_data()

        if train_classes is None and validate_classes is None and test_classes is None:
            rand = Random()
            rand.seed(1337)
            classes = list(self.__data.keys())
            rand.shuffle(classes)
            train_classes_count = int(0.8 * len(classes))
            train_classes = classes[:train_classes_count]
            validate_classes = classes[train_classes_count:]
            test_classes = classes[train_classes_count:]
        if test_classes is not None and validate_classes is not None:
            classes = list(self.__data.keys())
            train_classes = set(classes)
            train_classes -= set(test_classes)
            train_classes -= set(validate_classes)
            train_classes = list(train_classes)

        super().__init__(
            train_classes=train_classes, validate_classes=validate_classes, test_classes=test_classes,
            auto_load_data=True, return_1d_images=return_1d_audio_data, min_cluster_count=min_cluster_count,
            max_cluster_count=max_cluster_count
        )

    def _get_img_data_shape(self):
        return (self.__window_width, self.__audio_helper.get_default_spectrogram_coefficients_count(), 1)

    def __get_cache_key(self, path):
        return "[{}]_[{}]".format(self.__audio_helper.get_settings_str(), path)

    def __load_audio_file(self, path):
        cache_key = self.__get_cache_key(path)
        if self.__cache is not None and self.__cache.exists(cache_key):
            print("Load {} from the cache...".format(path))
            return self.__cache.load(cache_key)
        print("Load {}...".format(path))
        content = self.__audio_helper.audio_to_default_spectrogram(path)
        content = np.transpose(content)
        content = np.reshape(content, content.shape + (1,))
        if self.__cache is not None:
            self.__cache.save(cache_key, content)
        return content

    def _load_data(self):
        if self.__data is None:
            clusters = self._get_audio_file_clusters(self.__data_dir)
            output_clusters = {}
            t_start = time()
            for k in clusters.keys():

                # Get all audio snippets
                snippets = map(
                    lambda file: self.__load_audio_file(path.join(self.__data_dir, file)),
                    clusters[k]
                )

                # If required: Merge all snippets
                if self.__concat_audio_files_of_speaker:
                    snippets = [np.concatenate(list(snippets))]

                # Filter the snippets for the minimum length
                snippets = list(filter(
                    lambda snippet: snippet.shape[0] >= self.__window_width,
                    snippets
                ))

                if len(snippets) == 0:
                    print("WARNING: All input files for the class '{}' are to short (the length must be 1s or more). This class is ignored.".format(k))
                else:
                    output_clusters[k] = snippets
            t_end = time()
            t_delta = t_end - t_start
            print("Required {} seconds to load the data...".format(t_delta))
            self.__data = output_clusters
        return self.__data

    def _get_audio_file_clusters(self, data_dir):
        return {}

    def _get_random_element(self, class_name):
        audio_object = self.__rand.choice(self._get_data()[class_name])

        # Select a random snippet
        start_index_range = (0, audio_object.shape[0] - self.__window_width)
        start_index = self.__rand.randint(start_index_range[0], start_index_range[1] + 1)

        return audio_object[start_index:(start_index + self.__window_width)]

    def _image_plot_preprocessor(self, img):

        # Sometimes the range is not exactly [0, 1]; fix this
        img = np.minimum(img, 1.)
        img = np.maximum(img, 0.)

        # Swap the first two axes
        return img.swapaxes(0, 1)



