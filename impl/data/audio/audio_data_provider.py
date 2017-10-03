
# TODO
# Requirements:
# - Output can be a 2D image
# - Output can be a 1D image
# - Allow different audio data types (spectrogram types; maybe use the AUdioHelper from VT2 as an argument for the audio data provider)
import numpy as np
import random

from impl.data.image.image_data_provider import ImageDataProvider

from impl.data.misc.audio_helper import AudioHelper

from impl.misc.simple_file_cache import SimpleFileCache

class AudioDataProvider(ImageDataProvider):
    def __init__(self, data_dir=None, audio_helper=None, cache_directory=None, train_classes=None, validate_classes=None,
                 test_classes=None, auto_load_data=True, window_width=100, return_1d_audio_data=False,
                 min_cluster_count=None, max_cluster_count=None):
        super().__init__(
            train_classes=train_classes, validate_classes=validate_classes, test_classes=test_classes,
            auto_load_data=False, return_1d_images=return_1d_audio_data, min_cluster_count=min_cluster_count,
            max_cluster_count=max_cluster_count
        )

        self.__data_dir = data_dir

        if audio_helper is None:
            audio_helper = AudioHelper()
        self.__audio_helper = audio_helper
        self.__cache = None if cache_directory is None else SimpleFileCache(cache_directory)
        self.__window_width = window_width

        if auto_load_data:
            self.load_data()

    def __load_audio_file(self, path):
        if self.__cache is not None and self.__cache.exists(path):
            return self.__cache.load(path)
        content = self.__audio_helper.audio_to_default_spectrogram()
        if self.__cache is not None:
            self.__cache.save(path, content)
        return content

    def _load_data(self):
        clusters = self._get_audio_file_clusters(self.__data_dir)
        for k in clusters.keys():
            clusters[k] = list(filter(
                lambda obj: obj.shape[0] >= self.__window_width,
                map(
                    self.__load_audio_file,
                    clusters[k]
            )))

    def _get_audio_file_clusters(self, data_dir):
        return {}

    def _get_random_element(self, class_name):
        audio_object = np.random.choice(self._get_data()[class_name])

        # Select a random snippet
        start_index_range = (0, audio_object.shape[0] - self.__window_width)
        start_index = random.randint(start_index_range[0], start_index_range[1] + 1)

        return audio_object[start_index:(start_index + self.__window_width)]



