import matplotlib
matplotlib.use('Agg')

from impl.nn.base.cluster_nn.minimal_cluster_nn import MinimalClusterNN
from impl.nn.playground.cluster_nn_kl_divergence import ClusterNNKlDivergence

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.data.image.mnist_data_provider import MNISTDataProvider
    from impl.data.image.cifar10_data_provider import Cifar10DataProvider
    from impl.data.image.cifar100_data_provider import Cifar100DataProvider
    from impl.data.audio.timit_data_provider import TIMITDataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    dp = Simple2DPointDataProvider(min_cluster_count=2, max_cluster_count=3, allow_less_clusters=False)
    # dp = MNISTDataProvider(min_cluster_count=3, max_cluster_count=3)
    # dp = Cifar10DataProvider(min_cluster_count=3, max_cluster_count=3)
    # dp = Cifar100DataProvider(min_cluster_count=3, max_cluster_count=3)
    TIMIT20_lst = ['MTDB0','FCMG0','MABW0','MWEM0','MTLS0','MMAM0','MTJU0','FECD0','FVMH0','MDCD0','MJPG0','MRSP0','MRFK0','FCAU0','MRCG0','MRKM0','MPRT0','MCTT0','FEME0','MCRE0']
    dp = TIMITDataProvider(
        # data_dir=top_dir + "/test/TIMIT_mini", cache_directory=top_dir + "/test/cache",
        data_dir=top_dir + "/test/TIMIT_mini", cache_directory=top_dir + "/test/cache",
        min_cluster_count=1,
        max_cluster_count=3,
        return_1d_audio_data=False,
        # test_classes=TIMIT20_lst,
        # validate_classes=TIMIT20_lst,
        concat_audio_files_of_speaker=True
    )

    en = SimpleFCEmbedding()
    # en = None
    en = CnnEmbedding(block_feature_counts=[1, 2, 3], fc_layer_feature_counts=[4], output_size=3, dimensionality='auto')

    # cnn = MinimalClusterNN(dp, 5, en, weighted_classes=True)
    cnn = ClusterNNKlDivergence(dp, 5, en, weighted_classes=True)
    # cnn.class_weights_approximation = 'stochastic'
    # cnn.build_networks(print_summaries=True)
    cnn.build_networks(print_summaries=False)
    cnn.minibatch_size = 2
    cnn.validate_every_nth_epoch = 1

    # clusters = dp.get_data(50, 200)

    autosave_dir = top_dir + 'test/autosave_ClusterNN_playground'
    cnn.register_autosave(autosave_dir)  # , nth_iteration=1)

    cnn.train(1)

    # # clusters = dp.get_data(50, 200)
    #
    # c_nn = ClusterNNTry00(dp, 3, en, weighted_classes=True)
    #
    # # c_nn.f_cluster_count = lambda: 10
    # c_nn.minibatch_size = 2
    #
    # # i = 0
    # # start = time()
    # # while True:
    # #     try:
    # #         print(i)
    # #         c = dp.get_data(50, 200)
    # #         print("Min cluster count: {}, Max cluster count: {}".format(min(map(len, c)), max(map(len, c))))
    # #         now = time()
    # #         i += 1
    # #         print("Avg: {}".format((now - start) / i))
    # #     except:
    # #         print("ERROR")
    #
    # c_nn.build_networks()
    #
    # # Enable autosave and try to load the latest configuration
    # autosave_dir = top_dir + 'test/autosave_ClusterNN_playground'
    # c_nn.register_autosave(autosave_dir)#, nth_iteration=1)
    # c_nn.try_load_from_autosave(autosave_dir)
    #
    # # Train a loooong time
    # c_nn.train(1000000)


