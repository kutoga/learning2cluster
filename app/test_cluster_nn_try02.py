import matplotlib
matplotlib.use('Agg')

from impl.nn.try02.cluster_nn_try02 import ClusterNNTry02

if __name__ == '__main__':
    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "G:/tmp/"

    dp = Simple2DPointDataProvider()
    en = SimpleFCEmbedding()
    #en = None # DO NOT USE ANY embedding (just use the twodimensional-points)

    c_nn = ClusterNNTry02(dp, 50, en)
    c_nn.minibatch_size = 192

    c_nn.build_networks()

    # Enable autosave and try to load the latest configuration
    autosave_dir = top_dir + 'test/autosave_ClusterNNTry02'
    c_nn.register_autosave(autosave_dir)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


