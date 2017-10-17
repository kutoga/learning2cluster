from scipy.cluster.hierarchy import linkage
import numpy as np

def misclassification_rate(N, e):
    MR = float(e) / N
    return MR

def increase_error(indices, e, clusters):
    for i in indices:
        if i < len(e):
            e[i] = 1
        else:
            increase_error(clusters[i], e, clusters)


def calc_MR(X, y, linkage_metric):
    # cityblock, braycurtis,
    from scipy.spatial.distance import cdist
    X = cdist(X, X, linkage_metric)
    Z = linkage(X, method='complete', metric=linkage_metric)

    clusters = []
    for i in range(len(y)):
        clusters.append([i])

    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    e = []
    e.append(np.ones(len(y), dtype=np.int))
    for z in Z:
        err = list(e[len(e) - 1])
        idx1 = int(z[0])
        idx2 = int(z[1])
        if idx1 >= len(y) or idx2 >= len(y) or y[idx1] != y[idx2]:
            indices = clusters[idx1] + clusters[idx2]
            increase_error(indices, err, clusters)
        else:
            err[idx1] = 0
            err[idx2] = 0
        e.append(err)

    MRs = []
    for err in e:
        MRs.append(misclassification_rate(len(y), sum(err)))

    print('MR=%f' % np.min(MRs))
    return MRs

if __name__ == '__main__':
    X = np.asarray(([
        [0, 2], [1, 4], [2, 1], [3, 2]
    ]))
    calc_MR(X, [0, 0, 1, 1], 'euclidean')
