from os import makedirs


def try_makedirs(path):
    try:
        makedirs(path)
        return True
    except:
        return False

