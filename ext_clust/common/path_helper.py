"""
Single purpose file to determine the absolute path to the "common" folder.
"""
import inspect
import os.path as path


def get_common_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return path.dirname(path.abspath(filename))
