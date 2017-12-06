import matplotlib
matplotlib.use('Agg')

from impl.nn.try00.cluster_nn_try00 import ClusterNNTry00

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network uses weighted classes

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/cluster/home/meierbe8/data/MT/" if is_linux else "G:/tmp/"

    #fixedc = 5
    #dp = Simple2DPointDataProvider(min_cluster_count=fixedc, max_cluster_count=fixedc, allow_less_clusters=False)

    dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=10)

    #en = SimpleFCEmbedding()
    en = None

    c_nn = ClusterNNTry00(dp, 50, en, weighted_classes=True)

    # c_nn.f_cluster_count = lambda: 10
    c_nn.minibatch_size = 200
    c_nn.class_weights_approximation = 'stochastic'

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

    c_nn.build_networks()

    # Enable autosave and try to load the latest configuration
    autosave_dir = top_dir + 'test/autosave_ClusterNNTry00_noembd'
    c_nn.register_autosave(autosave_dir)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


