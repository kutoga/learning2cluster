import numpy as np


def uncertainity_measure(dist):
    """
    The uncertainity measure is just the normalized entropy of the input distribution.

    If the value is 0, then one single value is selected. If the value is 1 then the
    distribution is uniform

    This measure may be used for a neural network softmax output:
    If it is 0, then the network is quite sure about its selection. On the other side, if the value is 1,
    then the network is absolutely not sure about the selection.

    :param dist:
    :return:
    """
    num_classes = np.prod(dist.shape)

    # Simple case: 1 class => the (normalized) entropy is always 0
    if num_classes == 1:
        return 0.

    # Calculate the entropy
    non_zero_dist = dist[dist != 0.]
    entropy = -np.sum(non_zero_dist * np.log2(non_zero_dist))

    # Normalize it
    norm_entropy = entropy / np.log2(num_classes)

    return norm_entropy


def average_uncertainity_measure(dists):
    return np.mean(list(map(uncertainity_measure, dists)))


if __name__ == '__main__':
    print(uncertainity_measure(np.asarray([0.2, 0.6, 0.2])))
    print(uncertainity_measure(np.asarray([0.5, 0.5])))
    print(uncertainity_measure(np.asarray([1.0])))
