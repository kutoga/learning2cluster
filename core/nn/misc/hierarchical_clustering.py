import numpy as np

from ext_clust.common.analysis.analysis import cluster_embeddings, calculate_analysis_values, plot_curves

def hierarchical_clustering(x_values, true_clusters, cluster_nn=None, plot_filename=None):
    if cluster_nn is None:
        embeddings = x_values
    else:
        embeddings = cluster_nn.calculate_embeddings(x_values)
    n_embeddings = len(embeddings)

    # Do the clustering
    _, embeddings_linkage = cluster_embeddings(embeddings)
    mrs, homogeneity_scores, completeness_scores, thresholds = calculate_analysis_values(embeddings_linkage, true_clusters)

    # Create a plot if required
    if plot_filename is not None:
        plot_curves(plot_filename, ['test'], mrs, homogeneity_scores, completeness_scores, [n_embeddings])

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