import matplotlib
matplotlib.use('Agg')

from impl.nn.try00.cluster_nn_try00 import ClusterNNTry00

if __name__ == '__main__':
    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "G:/tmp/"

    dp = Simple2DPointDataProvider()
    en = SimpleFCEmbedding()

    c_nn = ClusterNNTry00(dp, 50, en)
    c_nn.minibatch_size = 200

    c_nn.build_networks()

    # Enable autosave and try to load the latest configuration
    autosave_dir = top_dir + 'test/autosave_ClusterNNTry00'
    c_nn.register_autosave(autosave_dir)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


