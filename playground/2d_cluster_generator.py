import numpy as np
import random

from sklearn.datasets import make_blobs, make_circles, make_moons

import matplotlib.pyplot as plt

def plot_data(data):

    cluster_indices = sorted(set(data[1]))
    clusters = [[] for c_i in cluster_indices]
    for i in range(len(data[1])):
        clusters[data[1][i]].append(data[0][i])

    fig, ax = plt.subplots()
    for cluster in clusters:
        px = np.asarray(list(map(lambda c: c[0], cluster)))
        py = np.asarray(list(map(lambda c: c[1], cluster)))
        ax.scatter(px, py)
    # plt.xlim(-0.2, 1.2)
    # plt.ylim(-0.2, 1.2)

    plt.show(block=True)

def rotate_2d(data, rad):
    c_r = np.cos(rad)
    s_r = np.sin(rad)
    r_m = np.asarray([[c_r, -s_r], [s_r, c_r]])
    return np.transpose(np.dot(r_m, np.transpose(data)))


def rescale_data(data, x_range, y_range):

    # Get all required values
    new_dx = x_range[1] - x_range[0]
    new_dy = y_range[1] - y_range[0]
    xmin = np.min(data[:, 0])
    xmax = np.max(data[:, 0])
    ymin = np.min(data[:, 1])
    ymax = np.max(data[:, 1])
    old_dx = xmax - xmin
    old_dy = ymax - ymin

    # And the just use simple math
    data[:, 0] = (data[:, 0] - xmin) * new_dx / old_dx + x_range[0]
    data[:, 1] = (data[:, 1] - ymin) * new_dy / old_dy + y_range[0]

    return data


def rand_bool():
    return bool(random.getrandbits(1))

# A cluster generator creates a cluster in the range [-1, 1]. It return a tuple (points, cluster_indices)
cluster_generators = []
cluster_generators.append(
    (2, lambda samples: make_circles(n_samples=samples, noise=random.uniform(0, 0.08), factor=random.uniform(0.1, 0.4)))
)

circles = make_circles(n_samples=15, noise=random.uniform(0, 0.08), factor=random.uniform(0.1, 0.4))

for i in range(10):
    data = make_blobs(50, centers=5)
    data = make_moons(n_samples=101, noise=random.uniform(0, 0.08))
    data_points = data[0]

    if rand_bool():
        data_points = -data_points
    data_points = rotate_2d(data_points, random.uniform(0, np.pi * 2))

    data_points = rescale_data(data_points, [0, 1], [0, 1])

    data = (data_points, data[1])

    plot_data(data)


# for i in range(10):
#     plot_data(cluster_generators[0](40))
