import matplotlib
matplotlib.use('Agg')

import numpy as np

from keras.layers.advanced_activations import LeakyReLU

from random import randint
from time import time

from impl.nn.try00.cluster_nn_try00_v30 import ClusterNNTry00_V30

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.image.flowers102_data_provider import Flowers102DataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "G:/tmp/"
    ds_dir = "./" if is_linux else "../"

    dp = Flowers102DataProvider(
        min_cluster_count=1,
        max_cluster_count=5,
    )
    en = CnnEmbedding(
        output_size=96,
        cnn_layers_per_block=2, block_feature_counts=[64, 128, 256],
        fc_layer_feature_counts=[512], hidden_activation=LeakyReLU(), final_activation=LeakyReLU(),
        batch_norm_for_init_layer=False, batch_norm_after_activation=True, batch_norm_for_final_layer=True,
        dropout_init=.5, dropout_after_max_pooling=[.5, .5], dropout_after_fc=[.5]
    )

    c_nn = ClusterNNTry00_V30(dp, 20, en, lstm_layers=7, internal_embedding_size=64, cluster_count_dense_layers=1, cluster_count_dense_units=256,
                              output_dense_layers=1, output_dense_units=256, cluster_count_lstm_layers=2, cluster_count_lstm_units=96,
                              kl_embedding_size=81)
    c_nn.include_self_comparison = False
    c_nn.weighted_classes = True
    c_nn.class_weights_approximation = 'stochastic'
    c_nn.minibatch_size = 40
    c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
    c_nn.validate_every_nth_epoch = 10

    # c_nn.f_cluster_count = lambda: 10
    # c_nn.minibatch_size = 200

    # c_nn._get_keras_loss()

    # i = 0
    # start = time()
    # while True:
    #     try:
    #         print(i)
    #         c = dp.get_data(50, 200)
    #         print("Min cluster count: {}, Max cluster count: {}".format(min(map(len, c)), max(map(len, c))))
    #         now = time()
    #         i += 1
    #         print("Avg: {}".format((now - start) / i))
    #     except:
    #         print("ERROR")

    c_nn.build_networks(print_summaries=False)

    # Enable autosave and try to load the latest configuration
    autosave_dir = top_dir + 'test/autosave_ClusterNNTry00_V33'
    c_nn.register_autosave(autosave_dir, example_count=15, nth_iteration=250, train_examples_nth_iteration=1000)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


