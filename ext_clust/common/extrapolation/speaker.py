"""
A Speaker contains all needed information and methods to create the pickle file used for training.

Based on previous work of Gerber, Lukic and Vogt, adapted by Heusser
"""
import pickle

import numpy as np

from ext_clust.common.spectogram.speaker_train_splitter import SpeakerTrainSplit
from ext_clust.common.spectogram.spectrogram_extractor import SpectrogramExtractor
from ext_clust.common.utils.paths import *


class Speaker:
    def __init__(self, split_train_test, max_speakers, speaker_list, output_name=None, sentences=10,
                 frequency_elements=128, max_audio_length=800):
        """
        Represents a fully defined speaker in the Speaker clustering suite.

        :param frequency_elements: How many frequency elements should be in a spectrogram
        :param split_train_test: Whether or not to split the test and training data
        :param max_speakers: The maximum amount of speakers to fetch from TIMIT
        :param speaker_list: The speaker list used to generate the pickle
        :param output_train: File name in which the train output pickle gets stored in
        :param output_test: File name in which the test output pickle gets stored in
        :param sentences: How many sentences there are from this speaker.
        :param max_audio_length: How long the audio of the speaker can maximally be
        """
        self.sentences = sentences
        self.frequency_elements = frequency_elements
        self.split_train_test = split_train_test
        self.max_speakers = max_speakers
        self.speaker_list = speaker_list
        self.max_audio_length = max_audio_length

        if output_name is None:
            self.output_name = speaker_list
        else:
            self.output_name = output_name

    def safe_to_pickle(self):
        """
        Fetches all data for this speaker from the TIMIT dataset and safes it inside of a pickle.
        """
        print("Extracting {}".format(self.speaker_list))

        X = np.zeros((self.max_speakers * 20, 1, self.frequency_elements, self.max_audio_length), dtype=np.float32)
        y = np.zeros(self.max_speakers * 20, dtype=np.int32)

        # Add all valid speakers
        valid_speakers = []
        with open(get_speaker_list(self.speaker_list), 'rb') as f:
            for line in f:
                # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                valid_speakers.append(bytes.decode(line.rstrip()))

        # Prepare SpectrogramExtractor
        extractor = SpectrogramExtractor(self.max_speakers, get_training("TIMIT"), valid_speakers)

        # Extract the spectrogram's, speaker numbers and speaker names
        X, y, speaker_names = extractor.extract_speaker_data(X, y)

        # Safe Test-Data to disk
        if self.split_train_test:
            speaker_train_split = SpeakerTrainSplit(0.2, self.sentences)
            X_train_valid, X_test, y_train_valid, y_test = speaker_train_split(X, y, None)

            with open(get_speaker_pickle(self.output_name + '_train'), 'wb') as f:
                pickle.dump((X_train_valid, y_train_valid, speaker_names), f, -1)

            with open(get_speaker_pickle(self.output_name + '_test'), 'wb') as f:
                pickle.dump((X_test, y_test, speaker_names), f, -1)
        else:
            with open(get_speaker_pickle(self.output_name + '_cluster'), 'wb') as f:
                pickle.dump((X, y, speaker_names), f, -1)

        print("Done Extracting {}".format(self.speaker_list))
        print("Safed to pickle.\n")

    def is_pickle_saved(self):
        """
        Whether or not a corresponding pickle file already exist.
        :return: true if it exists, false otherwise
        """
        if self.split_train_test:
            return path.exists(get_speaker_pickle(self.output_name + '_train'))
        else:
            path.exists(get_speaker_pickle(self.output_name + '_cluster'))
