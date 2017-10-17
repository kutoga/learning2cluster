# Requirements:
# - audio-format specifier (for the audio helper / loader)
# - path to the dataset

from os import path

from ast import literal_eval

from impl.data.audio.audio_data_provider import AudioDataProvider

class TIMITDataProvider(AudioDataProvider):
    def __init__(self, data_dir=None, audio_helper=None, cache_directory=None, window_width=100, return_1d_audio_data=False,
                 min_cluster_count=None, max_cluster_count=None, train_classes=None, validate_classes=None, test_classes=None,
                 concat_audio_files_of_speaker=False):
        super().__init__(
            data_dir=data_dir, audio_helper=audio_helper, cache_directory=cache_directory, window_width=window_width,
            return_1d_audio_data=return_1d_audio_data, min_cluster_count=min_cluster_count, max_cluster_count=max_cluster_count,
            train_classes=train_classes, validate_classes=validate_classes, test_classes=test_classes,
            concat_audio_files_of_speaker=concat_audio_files_of_speaker
        )

    def _get_audio_file_clusters(self, data_dir):
        list_files_file = path.join(data_dir, 'list_files.txt')
        with open(list_files_file, 'r') as fh:
            data = fh.read()
        return literal_eval(data)

    @staticmethod
    def load_speaker_list(path):

        # Load all lines
        with open(path, 'r') as fh:
            lines = [line.strip() for line in fh]

        # Filter the empty lines
        lines = list(filter(lambda line: line != "", lines))

        # Sort the lines
        lines = sorted(lines)

        # Thats it:)
        return lines

