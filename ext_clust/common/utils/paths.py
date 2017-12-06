"""
A bunch of path related convenience methods that avoid usage of relative paths for the application.
Uses the os.path library to use the correct path separators ( \\ or / ) automatically.
"""
import fnmatch
import os
import os.path as path

import ext_clust.common.path_helper
# import networks.path_helper as networks_helper


def join(base, *args):
    for arg in args:
        base = path.join(base, arg)

    return base


def get_common(*args):
    return join(ext_clust.common.path_helper.get_common_path(), *args)


# def get_networks(*args):
#     return join(networks_helper.get_networks_path(), *args)


# def get_configs(config):
#     """
#     Gets the absolute path to the config file of that name.
#     :param config: the name (without .cfg) of the file
#     :return: the absolute path of the config file
#     """
#     return get_networks('flow_me', 'config', config + '.cfg')


def get_data(*args):
    return join(get_common('data'), *args)


def get_experiments(*args):
    return join(get_data('experiments'), *args)


def get_experiment_logs(*args):
    return join(get_experiments('logs'), *args)


def get_experiment_nets(*args):
    return join(get_experiments('nets'), *args)


def get_experiment_plots(*args):
    return join(get_experiments('plots'), *args)


def get_experiment_cluster(*args):
    return join(get_experiments('clusters'), *args)


def get_speaker_list(speaker):
    """
    Gets the absolute path to the speaker list of that name.
    :param speaker: the name (without .txt) of the file
    :return: the absolute path of the speakerlist
    """
    return get_common('data', 'speaker_lists', speaker + '.txt')


def get_training(*args):
    return join(get_common('data', 'training'), *args)


def get_speaker_pickle(speaker):
    """
    Gets the absolute path to the speaker pickle of that name.
    :param speaker: the name (without .pickle) of the file
    :return: the absolute path of the speakers pickle
    """
    return get_training('TIMIT_extracted', speaker + '.pickle')


def get_results(*args):
    return join(get_data('results'), *args)


def get_result_pickle(network):
    """
    Gets the absolute path to the result pickle of that network.
    :param network: the name (without .pickle) of the file
    :return: the absolute path of the resut pickle
    """
    return join(get_results(), network + ".pickle")


def get_result_png(network):
    """
    Gets the absolute path to the result pickle of that network.
    :param network: the name (without .pickle) of the file
    :return: the absolute path of the resut pickle
    """
    return join(get_results(), network)


def list_all_files(directory, file_regex):
    """
    returns the absolute paths to the specified files.
    :param directory: the absolut path to de directory
    :param file_regex: a String that the files should match (fnmatch.fnmatch(file, file_regex))
    :return: the absolute path of al the files that match the file_regex an ar in the top level of the directory
    """
    files = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, file_regex):
            files.append(file)
    return files
