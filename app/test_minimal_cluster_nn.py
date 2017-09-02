import matplotlib
matplotlib.use('Agg')

from impl.nn.base.cluster_nn.minimal_cluster_nn import MinimalClusterNN

if __name__ == '__main__':
    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.nn.base.embedding_nn.simple_fc_embedding import SimpleFCEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/tmp/" if is_linux else "E:/tmp/"

    dp = Simple2DPointDataProvider(min_cluster_count=1, max_cluster_count=3)
    en = SimpleFCEmbedding()

    minimal_test = True

    if minimal_test:
        c_nn = MinimalClusterNN(dp, 3, en, weighted_classes=True)
        c_nn.minibatch_size = 1
        c_nn.validate_every_nth_epoch = 2
    else:
        c_nn = MinimalClusterNN(dp, 50, en)
        c_nn.minibatch_size = 200

    c_nn.build_networks()
    # c_nn.dummy_train() # Dummy training is required to load the history... (thats ugly)

    # Enable autosave and try to load the latest configuration
    autosave_dir = top_dir + 'test/autosave'
    if minimal_test:
        c_nn.register_autosave(autosave_dir, nth_iteration=1)
    else:
        c_nn.register_autosave(autosave_dir)
    c_nn.try_load_from_autosave(autosave_dir)

    c_nn.train(1000000)

    # try:
    #     c_nn.load_weights('E:/tmp/test/NN_WEIGHTS')
    # except:
    #     pass

    c_nn.save_plot(top_dir + 'test/history.png')

    # # Test if the prediction and training work
    # c_nn.dummy_predict()
    c_nn.dummy_train()

    # c_nn.test_network(2, 'E:/tmp/test')

    c_nn.save_weights(top_dir + 'test/NN_WEIGHTS')
