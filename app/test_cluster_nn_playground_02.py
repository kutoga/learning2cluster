import matplotlib
matplotlib.use('Agg')

import numpy as np

from keras.layers.advanced_activations import LeakyReLU

from random import randint
from time import time

from impl.nn.try04_ddbc.cluster_nn_try04_ddbc import ClusterNNTry04_Ddbc
from impl.nn.playground.cluster_nn_hint import ClusterNNHint
from impl.nn.playground.cluster_nn_merged_inputs import ClusterNNMergedInputs
from impl.nn.try00.cluster_nn_try00_v12 import ClusterNNTry00_V12
from impl.nn.try00.cluster_nn_try00_v16 import ClusterNNTry00_V16

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.image.mnist_data_provider import MNISTDataProvider
    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    # Create an mnist data provider that uses all images for tarining / testing. The neural network always returns 10 cluster options
    dp = MNISTDataProvider(
        train_classes=list(range(10)),
        test_classes=list(range(10)),
        validate_classes=list(range(10)),
        min_cluster_count=10,
        max_cluster_count=10
    )
    dp.target_min_cluster_count = 10
    dp.target_max_cluster_count = 10

    en = CnnEmbedding(
        output_size=100, cnn_layers_per_block=1, block_feature_counts=[32, 64],
        fc_layer_feature_counts=[], hidden_activation='relu', final_activation='relu',
        batch_norm_for_init_layer=True, cnn_filter_size=5
    )

    fixedc = 2
    dp = Simple2DPointDataProvider(
        min_cluster_count=fixedc, max_cluster_count=fixedc, allow_less_clusters=False
    )
    en = None
    # dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=10, allow_less_clusters=False)
    en = SimpleFCEmbedding(output_size=3, hidden_layers=[2], final_activation='tanh')

    c_nn = ClusterNNMergedInputs(dp, 3, en)
    c_nn.minibatch_size = 2
    c_nn.validate_every_nth_epoch = 1
    c_nn.debug_mode = True

    # c_nn.use_cluster_count_loss = False
    # c_nn.use_similarities_loss = False
    # c_nn.fixed_cluster_count_output = dp.get_max_cluster_count()

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

    c_nn.build_networks(print_summaries=False, build_training_model=False)

    c_nn.test_network(3, output_directory=top_dir + 'test/test_playground2')

    # # Enable autosave and try to load the latest configuration
    # autosave_dir = top_dir + 'test/autosave_ClusterNNPlayground02'
    # c_nn.register_autosave(autosave_dir, example_count=10, nth_iteration=1)
    # c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    # c_nn.train(1)

    y = c_nn.predict(
        np.asarray([
            [
                [0.2, 0.8],
                [0.1, 0.9],
                [0.3, 0.7]
            ],
            [
                [0.2, 0.8],
                [0.1, 0.9],
                [0.3, 0.7]
            ],
        ]),
        hints=[
            None,
            [[0, 1], [2]]
        ]
    )

    # Do a dummy prediction

