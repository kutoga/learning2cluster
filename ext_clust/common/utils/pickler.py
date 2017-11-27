import pickle


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
