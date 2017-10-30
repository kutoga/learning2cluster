from os import makedirs


def try_makedirs(path):
    try:
        makedirs(path)
        return True
    except:
        return False


def index_of(lst, obj, cmp=lambda x, y: x is y):
    for j in range(len(lst)):
        if cmp(lst[j], obj):
            return j
    return None

