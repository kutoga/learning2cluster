import numpy as np
from scipy.stats import entropy


# See:
# - Learning embeddings for speaker clustering based on voice equality
# - Unfolding Speaker Clustering Potential: A Biomimetic Approach
# - http://users.uoi.gr/cs01702/MargaritaKotti/MypublicationsPDFs/J3.pdf

def misclassification_rate_BV01(y_true, y_pred):
    """
    This MR is just an approximation (the real MR is always smaller or equal to the returned number, which means
    this implementation is in general pessimistic). The score is exact for relatively good clusterings and less
    exact for bad clusterings (it is still not that bad).

    :param y_true:
    :param y_pred:
    :return:
    """

    # First step: Find for each source cluster the target cluster:
    # This is the most critical step. We just use an approximation, because it seems to be very hard to get the optimal
    # cluster assignment.
    # 1) For each source cluster find the target cluster that contains the most elements of the source cluster. If the
    #    relative element count is >50%, then assign the target cluster to the source cluster. If there are multiple clusters
    #    for which this is true, then take the one with the most elements of the source cluster. If there are still multiple
    #    choices, then choose the cluster with the minimum element count (if there is such a single one).
    # 2) Order the non-assigned source clusters by their entropy (descending):
    #
    #      s = 0
    #      n = 0
    #      for each cluster c:
    #        c_n = len(c) # shared_elements_of_source_and(c)
    #        s += c_n * entropy(c) # Calculate a binary entropy: One set contains the elements which are in the current source cluster and the second all other elements
    #        n += c_n
    #      entropy = s / n
    #
    #    If two source clusters have the same entropy, order the clusters by the number of unassigned elements.
    #    If still two clusters get the same number, order them by their index (or some other deterministic number).
    # 3) Assign each source cluster to each target cluster with the highest relative element count. If there is no target
    #    cluster left with elements (or any target cluster) do not do any assignment.
    # Second step:
    # Now it is easy to calculate MR. We can easily calculate the number of correctly assigned elements (and therefore
    # also the number of wrong assigned elements).

    # Prepare the data
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape

    def build_cluster_dict(plain):
        indices = np.arange(plain.shape[0])
        return {ci: indices[plain == ci] for ci in np.unique(plain)}

    source_clusters = build_cluster_dict(y_true)
    target_clusters = build_cluster_dict(y_pred)
    unassigned_source_clusters = set(source_clusters.keys())
    unassigned_target_clusters = set(target_clusters.keys())

    # Find the cluster assignments
    cluster_assignments = {}
    # 1) Find the target cluster with the most elements
    for sci in list(unassigned_source_clusters):
        source_cluster = source_clusters[sci]

        # Search target clusters that match our criterion
        possible_target_clusters = []
        target_n_elements = None
        target_shared_elements = None

        for tci in unassigned_target_clusters:
            target_cluster = target_clusters[tci]
            target_elements = target_cluster.shape[0]
            shared_elements = np.intersect1d(source_cluster, target_cluster).shape[0]

            if (shared_elements / target_elements) > .5:
                if target_shared_elements is None or target_shared_elements < target_elements:
                    possible_target_clusters = [(tci, shared_elements)]
                    target_n_elements = target_elements
                    target_shared_elements = shared_elements
                elif target_shared_elements == shared_elements:
                    if target_n_elements > target_elements:
                        possible_target_clusters = [(tci, shared_elements)]
                        target_n_elements = target_elements
                        target_shared_elements = shared_elements
                    elif target_n_elements == target_elements:
                        possible_target_clusters.append((tci, shared_elements))

        # Check all possibilities:
        # 1) len(possible_target_clusters) == 0 => No assignment possible
        # 2) len(possible_target_clusters) == 1 => Am assignment is possible
        # 3) len(possible_target_clusters) > 1 => Try to choose the cluster with the highest relative element count

        if len(possible_target_clusters) == 1:
            tci = possible_target_clusters[0][0]
            cluster_assignments[sci] = tci
            unassigned_target_clusters.remove(tci)
            unassigned_source_clusters.remove(sci)
        elif len(possible_target_clusters) > 1:
            max_shared_elements = max(map(lambda x: x[1], possible_target_clusters))
            max_shared_elements_target_clusters = list(
                filter(lambda x: x[1] == max_shared_elements, possible_target_clusters))
            if len(max_shared_elements_target_clusters) == 1:
                # There is exactly one unique target cluster that has a maximum count of shared elements. Choose it.
                tci = max_shared_elements_target_clusters[0][0]
                cluster_assignments[sci] = tci
                unassigned_target_clusters.remove(tci)
                unassigned_source_clusters.remove(sci)

    # 2) Order the non-assigned source clusters by their entropy (descending) and then bei their cluster id (also descending)
    def source_entropy(sci):
        source_cluster = source_clusters[sci]
        s = 0
        n = 0
        for tci in unassigned_target_clusters:
            target_cluster = target_clusters[tci]
            target_elements = target_cluster.shape[0]
            shared_elements = np.intersect1d(source_cluster, target_cluster).shape[0]
            s += entropy([shared_elements, target_elements - shared_elements]) * target_elements
            n += target_elements
        if s == 0:
            return 0
        else:
            return s / n

    source_clusters_ordered = sorted(unassigned_source_clusters, reverse=True,
                                     key=lambda sci: (source_entropy(sci), sci))

    # 3) Assign each source cluster to each target cluster with the highest relative element count. If there is no target
    #    cluster left with elements (or any target cluster) do not do any assignement.
    for sci in source_clusters_ordered:
        source_cluster = source_clusters[sci]
        possible_target_clusters = []
        target_n_shared_relative = None
        for tci in unassigned_target_clusters:
            target_cluster = target_clusters[tci]
            target_elements = target_cluster.shape[0]
            shared_elements = np.intersect1d(source_cluster, target_cluster).shape[0]
            shared_elements_relative = shared_elements / target_elements
            if target_n_shared_relative is None or shared_elements_relative > target_n_shared_relative:
                possible_target_clusters = [tci]
                target_n_shared_relative = shared_elements_relative
            elif shared_elements_relative == target_n_shared_relative:
                possible_target_clusters.append(tci)
        if len(possible_target_clusters) == 1:
            tci = possible_target_clusters[0]
            cluster_assignments[sci] = tci
            unassigned_target_clusters.remove(tci)
            unassigned_source_clusters.remove(sci)
        elif len(possible_target_clusters) > 1:
            tci = sorted(possible_target_clusters)[0]
            cluster_assignments[sci] = tci
            unassigned_target_clusters.remove(tci)
            unassigned_source_clusters.remove(sci)

    # Second step:
    # Now it is easy to calculate MR. We can easily calculate the number of correctly assigned elements (and therefore
    # also the number of wrong assigned elements). This is the part which is described in the paper.
    sum_e_j = 0
    for sci in source_clusters.keys():
        source_cluster = source_clusters[sci]
        source_elements = source_cluster.shape[0]
        shared_elements = 0
        if sci in cluster_assignments:
            target_cluster = target_clusters[cluster_assignments[sci]]
            shared_elements = np.intersect1d(source_cluster, target_cluster).shape[0]
        e_j = source_elements - shared_elements
        sum_e_j += e_j
    MR = sum_e_j / y_true.shape[0]
    return MR


misclassification_rate = misclassification_rate_BV01

if __name__ == '__main__':
    def flt_eq(x, y):
        return abs(x - y) < 1e-5


    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 2, 2]
    assert flt_eq(misclassification_rate(y_true, y_pred), 0.0)

    y_true = [1, 1, 1, 1]
    y_pred = [2, 2, 1, 1]
    assert flt_eq(misclassification_rate(y_true, y_pred), 0.5)

    y_true = [1, 1, 1, 1]
    y_pred = [1, 2, 3, 4]
    assert flt_eq(misclassification_rate(y_true, y_pred), 0.75)

    y_true = [1, 2, 3, 4]
    y_pred = [1, 1, 1, 1]
    assert flt_eq(misclassification_rate(y_true, y_pred), 0.75)

    y_true = [1, 2, 2, 4]
    y_pred = [1, 1, 1, 1]
    assert flt_eq(misclassification_rate(y_true, y_pred), 0.5)

    y_true = [0, 0, 0, 1, 1, 1, 1, 2, 2]
    y_pred = [7, 7, 3, 3, 2, 2, 1, 0, 0]
    assert flt_eq(misclassification_rate_BV01(y_true, y_pred), 1 / 3)

    y_true = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert flt_eq(misclassification_rate_BV01(y_true, y_pred), 5 / 13)
