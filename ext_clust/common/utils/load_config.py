"""
    Loads and checks the config
    Based on previous work of Gygax and Egly.
"""
import configparser


def load_config(path_master_config, path_config):
    config = configparser.ConfigParser()

    # read master config file
    config.read_file(open(path_master_config))

    # read config file
    config.read_file(open(path_config))

    return config


def check_config(config):
    # create config checker
    # config_checker = ConfigChecker.ConfigChecker(config)

    # check config file
    # return config_checker.check_config_file()

    return True
