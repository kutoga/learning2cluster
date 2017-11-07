import matplotlib
matplotlib.use('Agg')
import random
import numpy as np

from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adadelta

from random import randint
from time import time

from impl.nn.try00.cluster_nn_try00_v51 import ClusterNNTry00_V51

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.audio.timit_data_provider import TIMITDataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"
    ds_dir = "./" if is_linux else "../"

    TIMIT_lst = TIMITDataProvider.load_speaker_list(ds_dir + 'datasets/TIMIT/traininglist_100/testlist_200.txt')

    # Shuffle the speakers
    random.seed(1)
    random.shuffle(TIMIT_lst)

    # Only use the first 20 speakers
    TIMIT_lst = TIMIT_lst[:20]

    dp = TIMITDataProvider(
        data_dir=top_dir + "/test/TIMIT", cache_directory=top_dir + "/test/cache",
        # data_dir=top_dir + "/test/TIMIT_mini", cache_directory=top_dir + "/test/cache",
        return_1d_audio_data=False,

        train_classes=TIMIT_lst,
        test_classes=TIMIT_lst,
        validate_classes=TIMIT_lst,
        min_cluster_count=1,
        max_cluster_count=5,

        concat_audio_files_of_speaker=True,

        # # Sample from these given window widths
        # window_width=[(80, 150)],
        # # window_width=[(40, 50), (70, 80)],
        #
        # # For each cluster we want at least one large snippet and one short snippet
        # minimum_snippets_per_cluster=[(200, 200), (50, 50)],
        #
        # split_audio_pieces_longer_than_and_create_hints=120

        window_width=200,
        minimum_snippets_per_cluster=2,
        snippet_merge_mode=[8,2]


        # minimum_snippets_per_cluster=[(200, 200), (100, 100)],
        # window_width=[(100, 200)]
    )
    dp.set_split_mode('train', 'snippet')
    dp.set_split_mode('test', 'snippet')

    required_input_count = dp.get_required_input_count_for_full_test('test')
    print("Required input count: {}".format(required_input_count))

    # data = dp.get_data(required_input_count, 1)

    for used_input_count in [10, 20, 40, 80, 160, required_input_count]:
        dp = TIMITDataProvider(
            data_dir=top_dir + "/test/TIMIT", cache_directory=top_dir + "/test/cache",
            # data_dir=top_dir + "/test/TIMIT_mini", cache_directory=top_dir + "/test/cache",
            return_1d_audio_data=False,

            train_classes=TIMIT_lst,
            test_classes=TIMIT_lst,
            validate_classes=TIMIT_lst,
            min_cluster_count=1,
            max_cluster_count=5,

            concat_audio_files_of_speaker=True,

            # # Sample from these given window widths
            # window_width=[(80, 150)],
            # # window_width=[(40, 50), (70, 80)],
            #
            # # For each cluster we want at least one large snippet and one short snippet
            # minimum_snippets_per_cluster=[(200, 200), (50, 50)],
            #
            # split_audio_pieces_longer_than_and_create_hints=120

            window_width=200,
            minimum_snippets_per_cluster=2,

            # Use this mode for the real avaluation
            snippet_merge_mode=[8, 2] if used_input_count == required_input_count else None

            # minimum_snippets_per_cluster=[(200, 200), (100, 100)],
            # window_width=[(100, 200)]
        )
        en = CnnEmbedding(
            output_size=256, cnn_layers_per_block=1, block_feature_counts=[32, 64, 128],
            fc_layer_feature_counts=[256], hidden_activation=LeakyReLU(), final_activation=LeakyReLU(),
            batch_norm_for_init_layer=False, batch_norm_after_activation=True, batch_norm_for_final_layer=True
        )

        c_nn = ClusterNNTry00_V51(dp, used_input_count, en, lstm_layers=7, internal_embedding_size=96, cluster_count_dense_layers=1, cluster_count_dense_units=256,
                                  output_dense_layers=1, output_dense_units=256, cluster_count_lstm_layers=1, cluster_count_lstm_units=128,
                                  kl_embedding_size=128, kl_divergence_factor=0.1)
        c_nn.include_self_comparison = False
        c_nn.weighted_classes = True
        c_nn.class_weights_approximation = 'stochastic'
        c_nn.minibatch_size = 35
        c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
        c_nn.set_loss_weight('similarities_output', 5.0)
        c_nn.optimizer = Adadelta(lr=5.0)

        validation_factor = 10
        c_nn.early_stopping_iterations = 10001
        c_nn.validate_every_nth_epoch = 10 * validation_factor
        c_nn.validation_data_count = c_nn.minibatch_size * validation_factor
        # c_nn.prepend_base_name_to_layer_name = False
        print_loss_plot_every_nth_itr = 100

        c_nn.build_networks(print_summaries=False, build_training_model=False)

        # Enable autosave and try to load the latest configuration
        autosave_dir_source = top_dir + 'test/autosave_ClusterNNTry00_V51'
        autosave_dir_target = top_dir + 'test/autosave_ClusterNNTry00_V64'

        # Load the best weights and create some examples
        c_nn.try_load_from_autosave(autosave_dir_source, config='best')

        c_nn.test_network(count=10, output_directory=autosave_dir_target + '/evaluation_{}'.format(used_input_count), data_type='test', create_date_dir=False)


