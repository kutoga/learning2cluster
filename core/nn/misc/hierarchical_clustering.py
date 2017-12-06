import numpy as np

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from ext_clust.common.analysis.analysis import cluster_embeddings, calculate_analysis_values, plot_curves

def hierarchical_clustering(x_values, true_clusters, cluster_nn=None, plot_filename=None, metric='cosine', method='complete'):
    if cluster_nn is None:
        embeddings = x_values
    else:
        embeddings = cluster_nn.calculate_embeddings(x_values)
    n_embeddings = len(embeddings)

    # Do the clustering
    _, embeddings_linkage = cluster_embeddings(embeddings, metric=metric, method=method)
    mrs, homogeneity_scores, completeness_scores, thresholds = calculate_analysis_values(embeddings_linkage, true_clusters)

    # Create a plot if required
    if plot_filename is not None:

        # TODO: Use this library plot (if it works)
        plot_curves(plot_filename, ['test'], [mrs], [homogeneity_scores], [completeness_scores], [n_embeddings])

        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        dendrogram(embeddings_linkage, ax=axes[0], above_threshold_color='y', orientation='top')
        axes[0].set_ylabel('threshold')
        axes[0].set_title('Dendrogram')

        axes[1].plot(thresholds, mrs, 'ro')
        axes[1].set_xlabel('threshold')
        axes[1].set_ylabel('MR')
        i = np.argmin(mrs)
        axes[1].set_title('Best threshold: {}, Best MR: {}'.format(thresholds[i], mrs[i]))

        plt.savefig(plot_filename + "_2.png")
        plt.clf()
        plt.close()

    return mrs, homogeneity_scores, completeness_scores, thresholds

if __name__ == '__main__':
    embeddings = [
        [1, 2],
        [3, 4],
        # [5, 6]
    ]
    true_clusters = [0, 0]
    mrs, homogeneity_scores, completeness_scores, thresholds = hierarchical_clustering(embeddings, true_clusters)
    print(mrs)
    pass