print(__doc__)

import numpy as np
import random

from scipy.cluster.vq import kmeans2

from itertools import chain

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from impl.data.simple_2d_point_data_provider import Simple2DPointDataProvider
from core.nn.misc.MR import misclassification_rate
from core.nn.misc.BBN import BBN

dp = Simple2DPointDataProvider(
    min_cluster_count=1,
    max_cluster_count=5,
    use_extended_data_gen=True,
    sigma=0.05
)
en = None

def generate_data(n=72):
    data, _, _ = dp.get_data(n, 1)
    data = data[0]

    cluster_count = len(data)
    print("Generated {} clusters...".format(cluster_count))

    x = np.asarray(list(chain(*data)))
    labels = list(chain(*map(lambda i: [i] * len(data[i]), range(len(data)))))

    return x, labels, cluster_count

def exec_test(fn_cluster, plot=False):

    # #############################################################################
    # Generate sample data
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                             random_state=0)

    X, labels_true, cluster_count = generate_data()

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    # db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_

    labels = fn_cluster(X, cluster_count)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    mr = misclassification_rate(labels_true, labels)

    mr = BBN(labels_true, labels, 0, True)

    print("NMI: {}".format(nmi))
    print("MR {}".format(mr))

    if plot:
        # #############################################################################
        # Plot result
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            # xy = X[class_member_mask & core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #          markeredgecolor='k', markersize=14)
            #
            # xy = X[class_member_mask & ~core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            #          markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show(block=True)

    return nmi, mr

def exec_tests(fn_cluster, n=300, plot=False):
    res = []
    for i in range(n):
        print("i={}".format(i))
        res.append(exec_test(fn_cluster, plot=plot))
        print()
    return res

n=300
# DBSCAN
res = exec_tests(
    lambda X, k: DBSCAN(eps=0.3, min_samples=10).fit(X).labels_,
    n=n
)
# k-means
# res = exec_tests(
#     # lambda X, k: kmeans2(X, max([1, k + random.randint(-1, 1)]))[1],
#     lambda X, k: kmeans2(X, k)[1],
#     n=n
# )
res = np.asarray(res)
nmi = res[:, 0]
mr = res[:, 1]

nmi_mean = np.mean(nmi)
nmi_std = np.std(nmi) / np.sqrt(n)

mr_mean = np.mean(mr)
mr_std = np.std(mr) / np.sqrt(n)

print("NMI: {} +/- {}".format(nmi_mean, nmi_std))
print("MR: {} +/- {}".format(mr_mean, mr_std))
