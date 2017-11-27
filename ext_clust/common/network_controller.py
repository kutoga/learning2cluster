"""
A NetworkController contains all knowledge and links to successfully train and test a network.

It's used to encapsulate a network and make use of shared code.
"""
import abc

from ext_clust.common.analysis.analysis import analyse_results
from ext_clust.common.utils.paths import get_speaker_pickle


class NetworkController:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.val_data = "speakers_40_clustering_vs_reynolds"
        self.name = name

    def get_validation_train_data(self):
        return get_speaker_pickle(self.val_data + "_train")

    def get_validation_test_data(self):
        return get_speaker_pickle(self.val_data + "_test")

    @abc.abstractmethod
    def train_network(self):
        """
        This method implements the training/fitting of the neural netowrk this controller implements.
        It handles the cycles, logging and saving.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_embeddings(self):
        """
        Processes the validation list and get's the embeddings as the network output.
        All return values are sets of possible multiples.
        :return: checkpoints, embeddings, speakers and the speaker numbers
        """
        return None, None, None, None

    def test_network(self):
        """
        Tests the network implementation with the validation data set and saves the result sets
        of the different metrics in analysis.
        """
        checkpoint_names, set_of_embeddings, set_of_speakers, speaker_numbers = self.get_embeddings()
        analyse_results(self.name, checkpoint_names, set_of_embeddings, set_of_speakers, speaker_numbers)
