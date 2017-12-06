from os import path
import random

import matplotlib.pyplot as plt

import numpy as np

from core.data.data_provider import DataProvider
from impl.data.misc.data_gen_2d import DataGen2dv02
from impl.data.misc.extended_data_gen_2d import ExtendedDataGen2d

class DummyDataProvider(DataProvider):
    def __init__(self, clusters, target_min_cluster_count=None, target_max_cluster_count=None):
        super().__init__(target_min_cluster_count=target_min_cluster_count, target_max_cluster_count=target_max_cluster_count)
        self.__clusters = self._convert_cluster_objs_to_numpy(clusters)

    def _convert_cluster_objs_to_numpy(self, clusters):
        return list(map(
            lambda cluster: list(map(
               lambda obj: np.asarray(obj, dtype=np.float32),
                cluster
            )),
            clusters
        ))

    def get_input_count(self):
        return sum(map(
            len, self.__clusters
        ))

    def get_min_cluster_count(self):
        return len(self.__clusters)

    def get_max_cluster_count(self):
        return len(self.__clusters)

    def get_data_shape(self):
        non_empty_clusters = list(filter(lambda cluster: len(cluster) > 0, self.__clusters))
        assert len(non_empty_clusters) > 0
        return non_empty_clusters[0][0].shape

    def _get_clusters(self, element_count, cluster_count=None, data_type='train'):
        """
        Generate some clusters and return them. Format [[obj1cluster1, obj2cluster1, ...], [obj1cluster2, ...]]
        :param element_count:
        :param cluster_count:
        :param test_data
        :return: clusters, additional_obj_info, clustering_hints
        """
        return self.__clusters, None, None
