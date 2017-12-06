import numpy as np

import matplotlib
matplotlib.use('Agg')

from impl.nn.try03_kmeans.cluster_nn_try03_kmeans_v04 import ClusterNNTry03KMeansV04

if __name__ == '__main__':
    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/cluster/home/meierbe8/data/MT/" if is_linux else "G:/tmp/"

    testing = False

    if testing:

        dp = Simple2DPointDataProvider(min_cluster_count=3, max_cluster_count=3)
        en = SimpleFCEmbedding(hidden_layers=0, output_size=2)
        en = None
        #en = None # DO NOT USE ANY embedding (just use the twodimensional-points)

        c_nn = ClusterNNTry03KMeansV04(
            dp, 3, en, lstm_layers=0, lstm_units=8, kmeans_itrs=2,
            cluster_count_dense_layers=0, cluster_count_dense_units=1,
            kmeans_input_dimension=2
        )
        c_nn.weighted_classes = True
        c_nn.minibatch_size = 2
        c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
        c_nn.validate_every_nth_epoch = 1

        c_nn.build_networks(print_summaries=False)

        # # Enable autosave and try to load the latest configuration
        autosave_dir = top_dir + 'test/autosave_ClusterNNTry03KMeansV04b'
        c_nn.register_autosave(autosave_dir, example_count=c_nn.minibatch_size)
        c_nn.try_load_from_autosave(autosave_dir)

        c_nn.additional_debug_array_printer = lambda a: a.tolist()
        c_nn.debug_mode = True

        # debug_outputs = []
        # c_nn.predict([
        #     [[0., 1.], [1., 0.], [0.7, 0.3]],
        #     [[0., 1.], [1., 0.], [0.7, 0.3]]
        # ], debug_mode=True, debug_outputs=debug_outputs)

        # Train a loooong time
        # c_nn.train(1)
        c_nn.test_network(4, top_dir + '/dummy')

    else:

        dp = Simple2DPointDataProvider(min_cluster_count=5, max_cluster_count=5)
        en = SimpleFCEmbedding(hidden_layers=[16, 32, 64, 32], output_size=8)
        #en = None
        # en = None # DO NOT USE ANY embedding (just use the twodimensional-points)

        c_nn = ClusterNNTry03KMeansV04(
            dp, 50, en, lstm_layers=0, lstm_units=32, kmeans_itrs=5,
            cluster_count_dense_layers=0, cluster_count_dense_units=1,
            kmeans_input_dimension=2
        )
        c_nn.weighted_classes = True
        c_nn.class_weights_approximation = 'stochastic'
        c_nn.minibatch_size = 200
        c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
        c_nn.validate_every_nth_epoch = 10

        c_nn.build_networks(print_summaries=False)

        # # Enable autosave and try to load the latest configuration
        autosave_dir = top_dir + 'test/autosave_ClusterNNTry03KMeansV04b'
        c_nn.register_autosave(autosave_dir)#, example_count=c_nn.minibatch_size)
        c_nn.try_load_from_autosave(autosave_dir)

        # c_nn.additional_debug_array_printer = lambda a: a.tolist()
        # c_nn.debug_mode = True

        # c_nn.predict([
        #     [[0., 1.], [1., 0.], [0.7, 0.3]]
        # ] * 2, debug_mode=True)

        # Train a loooong time
        c_nn.train(10000000)



