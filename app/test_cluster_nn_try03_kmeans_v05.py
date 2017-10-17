import numpy as np

import matplotlib
matplotlib.use('Agg')

from impl.nn.try03_kmeans.cluster_nn_try03_kmeans_v05 import ClusterNNTry03KMeansV05

if __name__ == '__main__':
    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    testing = False

    if testing:
        pass
    else:

        dp = Simple2DPointDataProvider(min_cluster_count=5, max_cluster_count=5)
        en = SimpleFCEmbedding(hidden_layers=[16, 32, 64, 32], output_size=8)
        #en = None
        # en = None # DO NOT USE ANY embedding (just use the twodimensional-points)

        c_nn = ClusterNNTry03KMeansV05(
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
        autosave_dir = top_dir + 'test/autosave_ClusterNNTry03KMeansV05'
        c_nn.register_autosave(autosave_dir)#, example_count=c_nn.minibatch_size)
        c_nn.try_load_from_autosave(autosave_dir)

        # c_nn.additional_debug_array_printer = lambda a: a.tolist()
        # c_nn.debug_mode = True

        # c_nn.predict([
        #     [[0., 1.], [1., 0.], [0.7, 0.3]]
        # ] * 2, debug_mode=True)

        # Train a loooong time
        c_nn.train(10000000)



