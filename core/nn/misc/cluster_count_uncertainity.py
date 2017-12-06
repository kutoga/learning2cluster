import numpy as np

import matplotlib.pyplot as plt

from os import path

from core.data.helper import progress
from core.helper import try_makedirs

def measure_cluster_count_uncertainity(network, nw_input, nw_hints=None, n_runs=1000, show_progress=False,
                                       output_directory=None, input_permutation=True, forward_pass_dropout=True):
    """
    Measure the uncertainity of the cluster count of the given network.
    The given input is used to measure it.

    To add nois eto the network one may enable forward_pass_dropout (defaultly enabled) or input_permutation.

    If the network is not yet built, it will be built. The following option is added:
    additional_build_config={
        'forward_pass_dropout': forward_pass_dropout
    }
    .

    :param network:
    :param nw_input:
    :return:
    """

    # Build the network if required
    if not network.is_network_built:
        network.build_networks(additional_build_config={
            'forward_pass_dropout': forward_pass_dropout
        })

    # Do all runs
    nw_input = np.asarray([nw_input])
    nw_hints = [nw_hints]
    results = []
    for i in range(n_runs):
        if show_progress:
            progress(i, n_runs)

        # If required: Permute the input
        if input_permutation:
            n = nw_input.shape[1]

            # TODO: Move this to an own function

            # First: Create a permutation
            p = np.random.permutation(list(range(n)))

            # 1) Permute the inputs
            nw_input[0] = nw_input[0][p]

            # 2) Permute the hints
            if nw_hints[0] is not None:
                from_to = list(range(n))[p]
                nw_hints[0] = [list(map(
                    lambda hint: tuple(map(lambda i: from_to[i], hint)),
                    nw_hints[0]
                ))]

        results.append(network.predict(nw_input, hints=nw_hints)[0])

    if show_progress:
        progress(n_runs, n_runs)

    # Collect the results in the following format:
    # {
    #    k_min: [0, 0.1, 0.2, 0.3, ...], # entries are sorted and are the probabilities for the given cluster count
    #    k_min+1: [0.2, 0.3, ...],
    #    ...
    #    k_max: [0.2, 0.4, ...]
    # }
    cluster_counts = list(network.data_provider.get_target_cluster_counts())
    stats = {
        k: [] for k in cluster_counts
    }
    for res in results:
        cluster_probabilities = res['cluster_count']
        for k in cluster_counts:
            stats[k].append(cluster_probabilities[k - cluster_counts[0]])
    for k in cluster_counts:
        stats[k].sort()

    try_makedirs(output_directory)

    # Create a bar char with 20 intervals and calculate a probability for each cluster count (probability=argmax(y))
    cluster_probabilities = {}
    steps = 20
    bar_chart_x = np.asarray(list(range(steps))) / steps + 1. / (2 * steps)
    for k in cluster_counts:
        bar_chart_y = [0.] * len(bar_chart_x)
        for p in stats[k]:
            if p == 1.:
                bar_chart_y[-1] += 1
            else:
                bar_chart_y[int(p * steps)] += 1
        bar_chart_y = np.asarray(bar_chart_y)
        bar_chart_y /= np.sum(bar_chart_y)
        cluster_probabilities[k] = bar_chart_x[np.argmax(bar_chart_y)]

        # Plot
        fig, ax = plt.subplots()

        # print("X: {}".format(x))
        # print("Y: {}".format(y))
        # print("Y_o: {}".format(distribution))

        ax.bar(bar_chart_x, bar_chart_y, 0.8 * 1 / steps, color="blue")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title("Cluster Count = {} (probability={})".format(k, cluster_probabilities[k]))

        if output_directory is None:
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.savefig(path.join(output_directory, '{:03d}_cluster_count_{}.png'.format(cluster_counts.index(k), k)))

        plt.clf()
        plt.close()

    # Create a resulting bar chart for the cluster count probabilities (they do not have to sum up to 1!)
    fig, ax = plt.subplots()
    ax.bar(cluster_counts, list(map(lambda k: cluster_probabilities[k], cluster_counts)), 0.9, color="blue")
    # plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("Cluster Probabilities")

    if output_directory is None:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.savefig(path.join(output_directory, '{:03d}_cluster_probabilities.png'.format(len(cluster_counts))))

    plt.clf()
    plt.close()

