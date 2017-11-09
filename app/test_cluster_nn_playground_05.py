import matplotlib
matplotlib.use('Agg')

from time import time

from impl.nn.base.cluster_nn.minimal_cluster_nn import MinimalClusterNN
from impl.nn.playground.cluster_nn_merged_inputs import ClusterNNMergedInputs
from impl.nn.playground.cluster_nn_kl_divergence import ClusterNNKlDivergence
from impl.nn.playground.cluster_nn_hint import ClusterNNHint
from impl.nn.playground.cluster_nn_merged_inputs import ClusterNNMergedInputs
from impl.nn.try00.cluster_nn_try00_v51 import ClusterNNTry00_V51

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.data.image.mnist_data_provider import MNISTDataProvider
    from impl.data.image.cifar10_data_provider import Cifar10DataProvider
    from impl.data.image.cifar100_data_provider import Cifar100DataProvider
    from impl.data.image.birds200_data_provider import Birds200DataProvider
    from impl.data.audio.timit_data_provider import TIMITDataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding
    from impl.nn.base.embedding_nn.cnn_bdlstm_embedding import CnnBDLSTMEmbedding
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding
    from impl.nn.base.embedding_nn.bdlstm_embedding import BDLSTMEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "G:/tmp/"
    ds_dir = "./" if is_linux else "../"

    dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=3, allow_less_clusters=False)
    en = SimpleFCEmbedding(hidden_layers=[2, 4, 4])


    autosave_dir = top_dir + 'test/autosave_ClusterNN_playground'

    # Define the possible input counts
    input_counts = [3, 4, 5]

    # Create an array with all possible models
    models = []

    master_network = None
    for input_count in input_counts:

        cnn = ClusterNNMergedInputs(dp, input_count, en, weighted_classes=True)
        models.append(cnn)

        # Share the state between all networks. This includes the history etc.
        # This must be done before the target network is built.
        if master_network is None:
            master_network = cnn
        else:
            cnn.share_state(master_network)

        # Build the network
        cnn.build_networks(print_summaries=False, build_training_model=True)
        cnn.minibatch_size = 2
        cnn.validate_every_nth_epoch = 1
        cnn.early_stopping_iterations = 5
        cnn.validation_data_count = cnn.minibatch_size * 5
        cnn.prepend_base_name_to_layer_name = False

        # Register autosave, try to load the latest weights and then start / continue the training
        cnn.register_autosave(autosave_dir, nth_iteration=1, example_count=1)

        # Just the master network has to load all weights
        if master_network == cnn:
            cnn.try_load_from_autosave(autosave_dir)


    # Do now the training: Use round robin
    while True:
        for nn in models:

            # Stop is early stopping was used
            if nn.train(1):
                break

    # Test all networks
    for nn in models:
        nn.test_network(count=6, output_directory=autosave_dir + '/examples_final_{}'.format(nn.input_count), data_type='test', create_date_dir=False)

