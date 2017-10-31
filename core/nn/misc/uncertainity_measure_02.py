import numpy as np


def uncertainity_measure_02(dist, norm=True):
    """
    The uncertainity measure 02 is the 1 minus the probability for
    the most probable choice of a distribution. It represents therefore
    the uncertainity of the decision of the neural network.

    If norm is True the value will be normalized to the range [0, 1].
    Without normalizing the range is [0, (n-1)/n], therefore the
    normalized result is just multiplied by n/(n-1)

    :param dist:
    :return:
    """
    n = np.prod(dist.shape)
    um = 1 - np.max(dist)
    if norm and n > 1:
        um *= n / (n - 1)
    return um


def average_uncertainity_measure_02(dists):
    return np.mean(list(map(uncertainity_measure_02, dists)))


if __name__ == '__main__':
    print(uncertainity_measure_02(np.asarray([0.2, 0.6, 0.2])))
    print(uncertainity_measure_02(np.asarray([0.2, 0.6, 0.2]), norm=False))
    print(uncertainity_measure_02(np.asarray([0.5, 0.5])))
    print(uncertainity_measure_02(np.asarray([0.5, 0.5]), norm=False))
    print(uncertainity_measure_02(np.asarray([1.0])))
    print(uncertainity_measure_02(np.asarray([1.0]), norm=False))
