import matplotlib
matplotlib.use('Agg')

from random import randint
from time import time

from impl.nn.try00.cluster_nn_try00 import ClusterNNTry00

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    dp = Simple2DPointDataProvider(min_cluster_count=2, max_cluster_count=3, allow_less_clusters=False)
    #en = SimpleFCEmbedding()
    en = None

    # clusters = dp.get_data(50, 200)

    c_nn = ClusterNNTry00(dp, 3, en, weighted_classes=True)

    # c_nn.f_cluster_count = lambda: 10
    c_nn.minibatch_size = 2

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
    autosave_dir = top_dir + 'test/autosave_ClusterNN_playground'
    c_nn.register_autosave(autosave_dir)#, nth_iteration=1)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


