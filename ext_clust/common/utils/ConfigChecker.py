"""
    Check if the config file is valid.
    Based on previous work of Gygax and Egly.
"""
import os


class ConfigChecker(object):
    def __init__(self, config):
        self.config = config

    def check_config_file(self):

        path = self.get_path('train', 'list')
        if not check_if_file_exists(path):
            not_exists(path, "File")
            return False

        path = self.get_path('train', 'pickle')
        if not check_if_file_exists(path):
            not_exists(path, "File")
            return False

        path = os.path.join(self.get_path('output', 'log_path'), 'train_' + self.get_path('exp', 'name'))
        if not check_if_dir_not_exists(path):
            not_exists(path, "Dir")
            return False

        path = os.path.join(self.get_path('output', 'log_path'), 'test_' + self.get_path('exp', 'name'))
        if not check_if_dir_not_exists(path):
            not_exists(path, "Dir")
            return False

        return True

    def get_path(self, category, specific):
        return self.config.get(category, specific)


def not_exists(path, kind):
    print(kind, ":", os.path._getfullpathname(path), "doesn't exist.")


def check_if_file_exists(path):
    return os.path.isfile(path)


def check_if_dir_not_exists(path):
    return not os.path.isdir(path)
