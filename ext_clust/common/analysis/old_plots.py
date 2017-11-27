import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn import manifold


def draw_dendogram(embeddings_linkage, embeddings_dist, labels, map_labels, map_labels_full, label_colors,
                   figsize=(6.4, 4.8)):
    """ Draws dendogramm with given linkage matrix from embeddings.
    """

    # Configuration of diagram
    plt.figure('Dendogram', figsize=figsize)
    plt.xlabel('speaker name', fontsize=11)
    plt.ylabel('distance', fontsize=11)
    plt.tick_params(labelsize=9)
    # matplotlib.rcParams['lines.linewidth'] = 4

    # leaf color
    leaf_colors = generate_leaf_colors(embeddings_linkage, embeddings_dist, label_colors, map_labels)

    dendrogram(embeddings_linkage, leaf_rotation=None, leaf_font_size=9, show_contracted=True, labels=map_labels_full,
               link_color_func=lambda x: leaf_colors[x])

    # Color label according to speaker
    x_axis = plt.gca()
    x_axis_labels = x_axis.get_xmajorticklabels()
    for x_axis_label in x_axis_labels:
        x_axis_label.set_color(label_colors[labels.index(x_axis_label.get_text())])

    plt.draw()


def generate_leaf_colors(embeddings_linkage, embeddings_dist, label_colors, map_labels):
    leaf_colors = ["#ffccff"] * (2 * len(embeddings_dist) - 1)
    clusters = []
    for speaker in map_labels:
        clusters.append([int(speaker)])

    for z in embeddings_linkage:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    for i in range(len(embeddings_linkage)):
        if int(embeddings_linkage[i, 0]) < len(map_labels) and int(embeddings_linkage[i, 1]) < len(map_labels) and \
                        map_labels[int(embeddings_linkage[i, 0])] == map_labels[int(embeddings_linkage[i, 1])]:
            leaf_colors[len(map_labels) + i] = label_colors[map_labels[int(embeddings_linkage[i, 0])]]
        elif len(set(clusters[len(map_labels) + i])) == 1:
            leaf_colors[len(map_labels) + i] = label_colors[clusters[len(map_labels) + i][0]]

    return leaf_colors


def draw_t_sne(embeddings, labels, map_labels, label_colors, legend=True):
    """ Draws t-SNE diagram with given embeddings.
    """
    # Calculate t-SNE
    tsne = manifold.TSNE(n_components=2, perplexity=50, early_exaggeration=1.0, learning_rate=100, metric="cosine",
                         init='pca', random_state=0)
    Y = tsne.fit_transform(embeddings)

    # Configuration of diagram
    plt.figure('t-SNE Plot')
    plt.xlabel('x', fontsize=11)
    plt.ylabel('y', fontsize=11)
    plt.tick_params(labelsize=9)

    # Sort array in order to plot the same speaker together
    order = np.argsort(map_labels)
    sorted_map_labels = map_labels[order]
    sorted_Y = Y[order]

    from_index = 0
    while from_index < len(map_labels):
        to_index = from_index
        while (to_index + 1) < len(map_labels) and sorted_map_labels[from_index] == sorted_map_labels[to_index + 1]:
            to_index += 1
        plt.scatter(sorted_Y[from_index:to_index + 1, 0], sorted_Y[from_index:to_index + 1, 1],
                    label=labels[sorted_map_labels[from_index]], c=label_colors[sorted_map_labels[from_index]])
        from_index = to_index + 1

    if legend:
        plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=8)

    plt.draw()
