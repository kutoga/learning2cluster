import matplotlib
matplotlib.use('Agg')

import numpy as np
from random import Random

import Augmentor
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adadelta

from random import randint
from time import time

from core.nn.misc.cluster_count_uncertainity import measure_cluster_count_uncertainity
from core.nn.misc.hierarchical_clustering import hierarchical_clustering

from impl.nn.try00.cluster_nn_try00_v122 import ClusterNNTry00_V122

if __name__ == '__main__':

    # Difference to test_cluster_nn_try00.py: No embedding is used and the network always returns that 10 clusters were
    # found, but some of them may be empty

    from sys import platform

    from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
    from impl.data.audio.timit_data_provider import TIMITDataProvider
    from impl.data.image.facescrub_data_provider import FaceScrubDataProvider
    from impl.data.image.birds200_data_provider import Birds200DataProvider
    from impl.nn.base.embedding_nn.cnn_embedding import CnnEmbedding

    is_linux = platform == "linux" or platform == "linux2"
    top_dir = "/cluster/home/meierbe8/data/MT_gpulab/" if is_linux else "G:/tmp/"
    ds_dir = "./" if is_linux else "../"

    p = Augmentor.Pipeline()
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.5)
    # p.flip_top_bottom(probability=0.5)
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)

    dp = Simple2DPointDataProvider(
        min_cluster_count=1,
        max_cluster_count=5,
        use_extended_data_gen=True,
        sigma=0.05
    )
    en = None

    def get_cnn():
        c_nn = ClusterNNTry00_V122(dp, 72, en, lstm_layers=14, internal_embedding_size=96*3, cluster_count_dense_layers=1, cluster_count_dense_units=256,
                                  output_dense_layers=0, output_dense_units=256, cluster_count_lstm_layers=1, cluster_count_lstm_units=128,
                                  kl_embedding_size=128, kl_divergence_factor=0.1, simplified_center_loss_factor=0.5)
        c_nn.include_self_comparison = False
        c_nn.weighted_classes = True
        c_nn.class_weights_approximation = 'stochastic'
        c_nn.minibatch_size = 200
        c_nn.class_weights_post_processing_f = lambda x: np.sqrt(x)
        c_nn.set_loss_weight('similarities_output', 5.0)
        c_nn.optimizer = Adadelta(lr=5.0)

        validation_factor = 10
        c_nn.early_stopping_iterations = 15001
        c_nn.validate_every_nth_epoch = 10 * validation_factor
        c_nn.validation_data_count = c_nn.minibatch_size * validation_factor
        # c_nn.prepend_base_name_to_layer_name = False
        return c_nn
    c_nn = get_cnn()

    c_nn.build_networks(print_summaries=False, build_training_model=False)

    # Load the configuration of try #01
    autosave_dir = top_dir + '/autosave_ClusterNNTry00_V125'
    # autosave_dir = top_dir + '/autosave_ClusterNNTry00_V125_try02'
    c_nn.try_load_from_autosave(autosave_dir)

    # Do some tests
    output_dir = top_dir + '/autosave_ClusterNNTry00_V128'
    c_nn.test_network(count=100, output_directory=output_dir + '/examples_final', data_type='test', create_date_dir=False)
    c_nn.test_network(count=300, output_directory=output_dir + '/examples_final_metrics', data_type='test', create_date_dir=False, only_store_scores=True)

