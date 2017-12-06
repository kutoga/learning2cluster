import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.metrics import *
from theano.gradient import np

from ext_clust.common.analysis.mr import misclassification_rate
from ext_clust.common.utils.logger import *
from ext_clust.common.utils.paths import *
from ext_clust.common.utils.pickler import load, save


def analyse_results(network_name, checkpoint_names, set_of_embeddings, set_of_speakers, speaker_numbers):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    set_of_mrs = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []
    set_of_thresholds = []

    for index, embeddings in enumerate(set_of_embeddings):
        logger.info('Analysing checkpoint:' + checkpoint_names[index])
        embeddings_distance, embeddings_linkage = cluster_embeddings(embeddings)

        mrs, homogeneity_scores, completeness_scores, thresholds = calculate_analysis_values(embeddings_linkage,
                                                                                             set_of_speakers[index])
        set_of_mrs.append(mrs)
        set_of_homogeneity_scores.append(homogeneity_scores)
        set_of_completeness_scores.append(completeness_scores)
        set_of_thresholds.append(thresholds)

    write_result_pickle(network_name, checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores,
                        set_of_completeness_scores, speaker_numbers)
    save_best_results(network_name, checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, speaker_numbers)
    logger.info('Analysis done')


def save_best_results(network_name, checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, speaker_numbers):
    if len(set_of_mrs) == 1:
        write_result_pickle(network_name + "_best", checkpoint_names, set_of_thresholds, set_of_mrs,
                            set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers)
    else:

        # Find best result (min MR)
        min_mrs = []
        for mrs in set_of_mrs:
            min_mrs.append(np.min(mrs))

        min_mr_over_all = min(min_mrs)

        best_checkpoint_name = []
        set_of_best_mrs = []
        set_of_best_homogeneity_scores = []
        set_of_best_completeness_scores = []
        set_of_best_thresholds = []
        best_speaker_numbers = []
        for index, min_mr in enumerate(min_mrs):
            if min_mr == min_mr_over_all:
                best_checkpoint_name.append(checkpoint_names[index])
                set_of_best_mrs.append(set_of_mrs[index])
                set_of_best_homogeneity_scores.append(set_of_homogeneity_scores[index])
                set_of_best_completeness_scores.append(set_of_completeness_scores[index])
                set_of_best_thresholds.append(set_of_thresholds[index])
                best_speaker_numbers.append(speaker_numbers[index])

        write_result_pickle(network_name + "_best", best_checkpoint_name, set_of_best_thresholds, set_of_best_mrs,
                            set_of_best_homogeneity_scores, set_of_best_completeness_scores, best_speaker_numbers)


def write_result_pickle(network_name, checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores,
                        set_of_completeness_scores, number_of_embeddings):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Write result pickle')
    save((checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores,
          number_of_embeddings), get_result_pickle(network_name))


def read_result_pickle(files):
    """
    Reads the results of a network from these files.
    :param files: can be 1-n files that contain a result.
    :return: curve names, thresholds, mrs, homogeneity scores, completeness scores and number of embeddings
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Read result pickle')
    curve_names = []

    # Initialize result sets
    set_of_thresholds = []
    set_of_mrs = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []
    set_of_number_of_embeddings = []

    # Fill result sets
    for file in files:
        curve_name, thresholds, mrs, homogeneity_scores, completeness_scores, number_of_embeddings = load(file)

        for index, threshold in enumerate(thresholds):
            set_of_thresholds.append(threshold)
            set_of_mrs.append(mrs[index])
            set_of_homogeneity_scores.append(homogeneity_scores[index])
            set_of_completeness_scores.append(completeness_scores[index])
            set_of_number_of_embeddings.append(number_of_embeddings[index])
            curve_names.append(curve_name[index])

    return curve_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores, set_of_number_of_embeddings


def plot_curves(plot_file_name, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings):
    """
    Plots all specified curves and saves the plot into a file.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Plot results')

    # How many lines to plot
    number_of_lines = len(curve_names)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, number_of_lines)]

    # Define number of figures
    fig1 = plt.figure(1)
    fig1.set_size_inches(32, 24)

    # Define Plots
    mr_plot = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    mr_plot.set_title('MR')
    mr_plot.set_xlabel('number of clusters')
    mr_plot.axis([0, 80, 0, 1])

    completeness_scores_plot = add_cluster_subplot(fig1, 223, 'completeness_scores')
    homogeneity_scores_plot = add_cluster_subplot(fig1, 224, 'homogeneity_scores')

    # Define curves and their values
    curves = [[mr_plot, mrs],
              [homogeneity_scores_plot, homogeneity_scores],
              [completeness_scores_plot, completeness_scores]]

    # Plot all curves
    for index in range(number_of_lines):
        label = curve_names[index]
        color = colors[index]
        number_of_clusters = np.arange(number_of_embeddings[index], 1, -1)

        for plot, value in curves:
            plot.plot(number_of_clusters, value[index], color=color, label=label)

        min_mr = np.min(mrs[index])
        mr_plot.annotate(str(min_mr), xy=(0, min_mr))

    # Add legend and save the plot
    fig1.legend()
    # fig1.show()
    fig1.savefig(get_result_png(plot_file_name))


def add_cluster_subplot(fig, position, title):
    """
    Adds a cluster subplot to the given figure.

    :param fig: the figure which gets a new subplot
    :param position: the position of this subplot
    :param title: the title of the subplot
    :return: the subplot itself
    """
    subplot = fig.add_subplot(position)
    subplot.set_title(title)
    subplot.set_xlabel('number of clusters')
    subplot.axis([0, 80, 0, 1])
    return subplot


def cluster_embeddings(embeddings, metric='cosine', method='complete'):
    """
    Calculates the distance and the linkage matrix for these embeddings.

    :param embeddings: The embeddings we want to calculate on
    :param metric: The metric used for the distance and linkage
    :param method: The linkage method used.
    :return: The embedding Distance and the embedding linkage
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Cluster embeddings')

    embeddings_distance = cdist(embeddings, embeddings, metric)
    embeddings_linkage = linkage(embeddings_distance, method, metric)

    return embeddings_distance, embeddings_linkage


def calculate_analysis_values(embeddings_linkage, true_clusters):
    """
    Calculates the analysis values out of the embedding linkage.

    :param embeddings_linkage: The linkage we calculate the values for.
    :param true_clusters: The validation clusters
    :return: misclassification rate, homogeneity Score, completeness score and the thresholds.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')

    thresholds = embeddings_linkage[:, 2]
    threshold_shape = thresholds.shape

    # Initialize output
    mrs = np.ones(threshold_shape)
    homogeneity_scores = np.ones(threshold_shape)
    completeness_scores = np.ones(threshold_shape)

    # Loop over all possible clustering
    for i, threshold in enumerate(thresholds):
        predicted_clusters = fcluster(embeddings_linkage, threshold, 'distance')

        # Calculate different analysis's
        mrs[i] = misclassification_rate(true_clusters, predicted_clusters)
        homogeneity_scores[i] = homogeneity_score(true_clusters, predicted_clusters)
        completeness_scores[i] = completeness_score(true_clusters, predicted_clusters)

    return mrs, homogeneity_scores, completeness_scores, thresholds


def read_and_safe_best_results():
    checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers = read_result_pickle(
        [get_result_pickle('flow_me')])
    save_best_results('flow_me', checkpoint_names, set_of_thresholds, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, speaker_numbers)


if __name__ == '__main__':
    read_and_safe_best_results()
