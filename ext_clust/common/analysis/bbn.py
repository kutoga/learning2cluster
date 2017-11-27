import numpy as np


# This file implements the BBN metrics that is described by Kotti et al.:
# https://spiral.imperial.ac.uk/bitstream/10044/1/11711/2/SP_Elsevier_2008_Margarita_Kotti.pdf
# (page 28)

def BBN(y_true, y_pred, Q=0., normalize=False):
    """
    The BBN metric

    :param y_true:
    :param y_pred:
    :param Q:
    :param normalize: The normalization is not defined in the original paper and is only fully defined if Q==0.
                    If Q>0 it may be the case that the score returns "None" (because of a zero-division).
    :return:
    """

    def create_cluster_dicts(clustering):
        if not isinstance(clustering, list):
            clustering = list(clustering)
        cluster2index = {c: set() for c in np.unique(clustering)}
        for i in range(len(clustering)):
            cluster2index[clustering[i]].add(i)
        return cluster2index

    # Create dicitonaries that contain for each cluster the list of indices
    true_c2i = create_cluster_dicts(y_true)
    pred_c2i = create_cluster_dicts(y_pred)

    # Calculate now the sum of the paper:
    # Remember, y_true describes the true speakers and y_pred the clustering.
    clusters = sorted(pred_c2i.keys())
    Nc = len(clusters)
    s = 0.
    for c_i in pred_c2i.values():
        for s_j in true_c2i.values():
            n_ij = len(c_i & s_j)
            n_i = len(c_i)
            s += (n_ij ** 2) / n_i
    s -= Q * Nc

    # The normalization is just an extension done by me (Benjamin Meier). It is not defined anywhere
    if normalize:

        # Hehe, thats the easiest way to find the maximum value;-)
        # Just calculate the score for the case y_pred:=y_true (this returns the maximum possible value).
        # There is just one problem: If Q is not equal to 0, the score may also include negative values. A normalization
        # currently doesn't work for Q!=0, because sometimes the maximum value is 0
        max_v = BBN(y_true, y_true, Q=Q, normalize=False)

        if max_v == 0:
            s = None
        else:
            s /= max_v

    return s


if __name__ == '__main__':
    def test(y_true, y_pred):
        print("y_true={}".format(y_true))
        print("y_pred={}".format(y_pred))
        print("BBN[Q={}]={}".format(0., BBN(y_true, y_pred, Q=0.)))
        print("BBN[Q={}]_normalized={}".format(0., BBN(y_true, y_pred, Q=0., normalize=True)))
        print("BBN[Q={}]={}".format(1., BBN(y_true, y_pred, Q=1.)))
        print("BBN[Q={}]_normalized={}".format(1., BBN(y_true, y_pred, Q=1., normalize=True)))
        print()


    test([1, 2, 3], [4, 5, 6])
    test([0, 0, 1, 1], [2, 2, 3, 2])
    test([0, 2, 1, 1], [2, 2, 3, 2])
