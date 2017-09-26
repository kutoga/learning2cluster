import matplotlib
matplotlib.use('Agg')

import numpy as np

from random import randint
from time import time

from impl.nn.try00.cluster_nn_try00_v02 import ClusterNNTry00_V02

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    fixedc = 7
    dp = Simple2DPointDataProvider(
        min_cluster_count=fixedc, max_cluster_count=fixedc, allow_less_clusters=False
    )
    # dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=10, allow_less_clusters=False)
    en = SimpleFCEmbedding(output_size=2, hidden_layers=[16, 32, 64, 64], final_activation='tanh')
    # en = None

    c_nn = ClusterNNTry00_V02(dp, 50, en, lstm_layers=5, lstm_units=64, cluster_count_dense_layers=1, cluster_count_dense_units=128,
                          output_dense_layers=1, output_dense_units=128)
    c_nn.weighted_classes = True
    c_nn.class_weights_approximation = 'stochastic'
    c_nn.minibatch_size = 200
    c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
    c_nn.validate_every_nth_epoch = 10

    # c_nn.f_cluster_count = lambda: 10
    c_nn.minibatch_size = 200

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
    autosave_dir = top_dir + 'test/autosave_ClusterNNTry00_V02'
    c_nn.register_autosave(autosave_dir)#, nth_iteration=1)
    c_nn.try_load_from_autosave(autosave_dir)

    # Train a loooong time
    c_nn.train(1000000)


