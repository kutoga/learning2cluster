import matplotlib
# matplotlib.use('Agg')

import numpy as np

from keras.layers.advanced_activations import LeakyReLU

from random import randint
from time import time

from impl.nn.try04_ddbc.cluster_nn_try04_ddbc import ClusterNNTry04_Ddbc
from impl.nn.playground.cluster_nn_simple_center_loss import ClusterNNSimpleCenterLoss
from impl.nn.playground.cluster_nn_forward_pass_dropout import ClusterNNForwardPassDropout
from impl.nn.try00.cluster_nn_try00_v12 import ClusterNNTry00_V12
from impl.nn.try00.cluster_nn_try00_v16 import ClusterNNTry00_V16


from core.nn.helper import loss_rand_index, loss_fowlkes_mallows

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.image.mnist_data_provider import MNISTDataProvider
    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.data.dummy_data_provider import DummyDataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "G:/tmp/"

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
        min_cluster_count=fixedc, max_cluster_count=fixedc,
    )
    en = None
    en = SimpleFCEmbedding(2)

    dp = DummyDataProvider([
        [
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        [
            [0.3, 0.7],
            [0.2, 0.7],
            [0, 1],
        ],
        [
            [0, 1],
            [2, 3]
        ]
    ])

    dp = Simple2DPointDataProvider(
        min_cluster_count=1, max_cluster_count=3,
    )

    # dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=10, allow_less_clusters=False)
    # en = SimpleFCEmbedding(output_size=2, hidden_layers=[16, 32, 64, 64], final_activation='tanh')

    c_nn = ClusterNNForwardPassDropout(dp, 3, en)
    c_nn.minibatch_size = 2
    c_nn.validate_every_nth_epoch = 1
    # c_nn.debug_mode = True
    c_nn.include_self_comparison = False

    # c_nn.use_cluster_count_loss = False
    # c_nn.use_similarities_loss = False
    c_nn.fixed_cluster_count_output = dp.get_max_cluster_count()


    # c_nn.build_networks(print_summaries=False, additional_build_config={
    #     'forward_pass_dropout': True
    # })

    # # Enable autosave and try to load the latest configuration
    # autosave_dir = top_dir + 'test/autosave_ClusterNNPlayground02'
    # c_nn.register_autosave(autosave_dir, example_count=10, nth_iteration=1)
    # c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    # c_nn.train(5)

    # y = c_nn.predict(
    #     np.asarray([
    #         [
    #             [0.2, 0.8],
    #             [0.1, 0.9],
    #             [0.3, 0.7]
    #         ]
    #     ]),
    #     hints=[
    #         None
    #     ]
    # )


    # Create the forward pass analysis
    from core.nn.misc.cluster_count_uncertainity import measure_cluster_count_uncertainity
    # x = measure_cluster_count_uncertainity(c_nn, [
    #             [0.2, 0.8],
    #             [0.1, 0.9],
    #             [0.3, 0.7]
    #         ], show_progress=True, input_permutation=True, forward_pass_dropout=True)
    x = measure_cluster_count_uncertainity(c_nn, [
                [0.2, 0.8],
                [0.1, 0.9],
                [0.3, 0.7]
            ], show_progress=True, output_directory='G:/tmp/test/measure_cluster_count_uncertainity',
            input_permutation=True, forward_pass_dropout=True)

    # Hierarchical Clustering
    from core.nn.misc.hierarchical_clustering import hierarchical_clustering

    # Generate data
    records = 50
    data, _, _ = c_nn.data_provider.get_data(elements_per_cluster_collection=records, data_type='test', cluster_collection_count=1)
    x_data, _ = c_nn._build_Xy_data(data, ignore_length=True)
    i_data = c_nn.data_to_cluster_indices(data)

    # Only use the first cluster collection
    x_data = list(map(lambda x: x[0], x_data[:-1]))
    i_data = i_data[0]

    mrs, homogeneity_scores, completeness_scores, thresholds = hierarchical_clustering(
        x_data, i_data, c_nn, plot_filename='G:/tmp/test/measure_cluster_count_uncertainity/out.png'
    )

    print(mrs)
    pass

    # Do a dummy prediction

