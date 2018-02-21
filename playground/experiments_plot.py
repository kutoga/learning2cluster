import matplotlib.pyplot as plt
import pickle
import os

from core.nn.helper import sliding_window_average

# source_file = "G:/tmp/test/gpulogin.cloudlab.zhaw.ch/MT_gpulab/autosave_ClusterNNTry00_V100/NN_ClusterNNTry00_V98_autosave_best_NN_ClusterNNTry00_V98_NN_ClusterNNTry00_V98_cluster_nn.history.pkl"
# output_diretory = 'G:/tmp/plots'

base_dir = 'G:/tmp/test/gpulogin.cloudlab.zhaw.ch/MT_gpulab/'
def get_dir(path):
    return os.path.join(base_dir, path)

jobs = [{
    'output_directory': 'G:/tmp/plots',
    'tasks': [
        # {
        #     'name': 'test',
        #     'source_file': 'G:/tmp/test/gpulogin.cloudlab.zhaw.ch/MT_gpulab/autosave_ClusterNNTry00_V100/NN_ClusterNNTry00_V98_autosave_itr_NN_ClusterNNTry00_V98_NN_ClusterNNTry00_V98_cluster_nn.history.pkl',
        # }
    ]
}, {
    'output_directory': 'G:/tmp/experiments_plots',
    'tasks': [{
        'name': '000_ex_2d_points',
        'source_file': get_dir('autosave_ClusterNNTry00_V130/NN_ClusterNNTry00_V122_autosave_itr_NN_ClusterNNTry00_V122_NN_ClusterNNTry00_V122_cluster_nn.history.pkl'),
    },
    #     {
    #     'name': '001_ex_timit',
    #     'source_file': get_dir('autosave_ClusterNNTry00_V122/NN_ClusterNNTry00_V122_autosave_itr_NN_ClusterNNTry00_V122_NN_ClusterNNTry00_V122_cluster_nn.history.pkl'),
    # }, {
    #     'name': '002_ex_coil_100',
    #     'source_file': get_dir('autosave_ClusterNNTry00_V136/NN_ClusterNNTry00_V122_autosave_itr_NN_ClusterNNTry00_V122_NN_ClusterNNTry00_V122_cluster_nn.history.pkl'),
    # }, {
    #     'name': '003_ex_facescrub',
    #     'source_file': get_dir('autosave_ClusterNNTry00_V126/NN_ClusterNNTry00_V122_autosave_itr_NN_ClusterNNTry00_V122_NN_ClusterNNTry00_V122_cluster_nn.history.pkl'),
    # }, {
    #     'name': '004_ex_tiny_imagenet',
    #     'source_file': get_dir('autosave_ClusterNNTry00_V129/NN_ClusterNNTry00_V122_autosave_itr_NN_ClusterNNTry00_V122_NN_ClusterNNTry00_V122_cluster_nn.history.pkl'),
    # }, {
    #     'name': '005_ex_facescrub_ca',
    #     'source_file': get_dir('autosave_ClusterNNTry00_V139/NN_ClusterNNTry00_V135_autosave_itr_NN_ClusterNNTry00_V135_NN_ClusterNNTry00_V135_cluster_nn.history.pkl'),
    # },
    ]
}]

def try_makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

for job in jobs:
    base_directory = job['output_directory']
    try_makedirs(base_directory)
    for task in job['tasks']:
        # output_directory = os.path.join(base_directory, task['name'])
        output_directory = base_directory + '/' + task['name']
        source_file = task['source_file']

        try:
            with open(source_file, 'rb') as fh:
                history = pickle.load(fh)
        except:
            print("Failed to open '{}'".format(source_file))
            continue

        loss_plots = [
            # {
            #     'fname': 'loss',
            #     'title': 'Loss',
            #     'data': [
            #         ('Loss: Training', 'loss'),
            #         ('Loss: Validation', 'val_loss')
            #     ]
            # },
            {
                'fname': 'loss_terms',
                'title': 'Loss Terms',
                'data': [
                    ('Total Loss: Training', 'loss'),
                    # ('Total Loss: Validation', 'val_loss'), # {'plot_min': True}),
                    ('Cluster Assignment ($P_{ij}$) Loss: Training', 'similarities_output_loss'),
                    # ('Cluster Assignment ($P_{ij}$) Loss: Validation', 'val_similarities_output_loss'),
                    ('Cluster Count Loss: Training', 'cluster_count_output_loss'),
                    # ('Cluster Count Loss: Validation', 'val_cluster_count_output_loss'),
                ],
                'legend_loc': 'upper right'
            }
        ]

        metric_plots = [
            {
                'fname': 'accuracy',
                'title': 'Accuracy',
                'data': [
                    ('Cluster Count $k$: Training', 'cluster_count_output_categorical_accuracy'),
                    # ('Cluster Count $k$: Validation', 'val_cluster_count_output_categorical_accuracy'),
                    # ('Cluster Assignment ($P_{ij}$): Training', 'similarities_output_acc'),
                    # ('Cluster Assignment ($P_{ij}$): Validation', 'val_similarities_output_acc'),
                ]
            },
            {
                'fname': 'metrics',
                'title': 'Metrics',
                'data': [
                    ('MR', 'metric_misclassification_rate_BV01'),
                    ('NMI', 'metric_normalized_mutual_info_score'),
                    ('BBN$_{\mathrm{norm}}$', 'metric_bbn_q0_normalized'),

                    # ('Completeness Score', 'metric_completeness_score'),
                    # ('Adjusted Mutual Info', 'metric_adjusted_mutual_info_score'),
                    # ('Adjusted Rand Score', 'metric_adjusted_rand_score'),
                    # ('Fowlkes Mallows', 'metric_fowlkes_mallows_score'),
                    # ('V-Measure', 'metric_v_measure_score'),
                    # ('Purity', 'metric_purity_score'),
                ]
            },
            {
                'fname': 'mr',
                'title': 'MR',
                'data': [
                    ('MR', 'metric_misclassification_rate_BV01')
                ]
            },
            {
                'fname': 'nmi',
                'title': 'NMI',
                'data': [
                    ('NMI', 'metric_normalized_mutual_info_score')
                ]
            },
            {
                'fname': 'bbn_norm',
                'title': 'BBN$_{\mathrm{norm}}$',
                'data': [
                    ('BBN$_{\mathrm{norm}}$', 'metric_bbn_q0_normalized')
                ]
            }
        ]

        def window_range(data):
            return int(0.0025 * len(data))

        # try_makedirs(output_directory)
        plot_id = 0

        show_figs = False

        def get_plot_filename(name, ftype='pdf'):
            global plot_id
            # res = os.path.join(output_directory, '{:03d}_{}.{}'.format(plot_id, name, ftype))
            res = output_directory + '_{:03d}_{}.{}'.format(plot_id, name, ftype)
            plot_id += 1
            return res

        # Create all loss plots
        for lplot in loss_plots:
            plot_points = []

            labels = []
            plt.figure(1, figsize=(9, 3))
            plt.grid(b=True)#, which='minor')
            for datar in lplot['data']:
                if len(datar) == 2:
                    label, key = datar
                    config = {}
                else:
                    label, key, config = datar
                data = history[key]
                x = list(range(len(data)))
                y = sliding_window_average(data, window_range=window_range(data))
                plt.plot(x, y, alpha=0.7, lw=1.0)
                labels.append(label)
                if 'plot_min' in config and config['plot_min']:
                    y_min = min(filter(lambda p: p is not None, data))
                    x_min = data.index(y_min)
                    plot_points.append((x_min, y_min))
            plt.title(lplot['title'])
            plt.legend(labels, loc=lplot['legend_loc'] if 'legend_loc' in lplot else 'best')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')

            if len(plot_points) > 0:
                plt.plot(
                    list(map(lambda p: p[0], plot_points)),
                    list(map(lambda p: p[1], plot_points)),
                    'o'
                )

            plt.tight_layout()
            plt.show(block=show_figs)

            fname = get_plot_filename(lplot['fname'])
            plt.savefig(fname)
            plt.savefig(fname + '.png')

            plt.clf()
            plt.close()

        # Create all metric plots
        for lplot in metric_plots:
            labels = []
            plt.figure(1, figsize=(9, 3))
            plt.grid(b=True)#, which='minor')
            for label, key in lplot['data']:
                data = history[key]
                x = list(range(len(data)))
                y = sliding_window_average(data, window_range=window_range(data))
                plt.plot(x, y, alpha=0.7, lw=1.0)
                labels.append(label)
            plt.title(lplot['title'])
            plt.legend(labels)
            plt.xlabel('Iteration')
            plt.ylabel(lplot['title'])
            plt.ylim([-0.05, 1.05])
            plt.tight_layout()
            plt.show(block=show_figs)

            fname = get_plot_filename(lplot['fname'])
            plt.savefig(fname)
            plt.savefig(fname + '.png')

            plt.clf()
            plt.close()

        pass