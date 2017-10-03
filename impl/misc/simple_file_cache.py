from os import path
import hashlib
import pickle
import shutil

from core.helper import try_makedirs


class SimpleFileCache:
    def __init__(self, cache_dir=None):
        self.__cache_dir = cache_dir
        try_makedirs(self.__cache_dir)

    def __md5sum(self, str):
        md5 = hashlib.md5()
        md5.update(str)
        return md5.hexdigest()

    def __get_filename(self, key):
        return path.join(
            self.__cache_dir,
            '.{}.pkl'.format(key)
        )

    def __write_obj(self, file, obj):
        with open(file, 'wb') as fh:
            pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)

    def __read_obj(self, file):
        with open(file, 'rb') as fh:
            return pickle.load(fh)

    def clear(self):
        shutil.rmtree(self.__cache_dir)
        try_makedirs(self.__cache_dir)

    def remove(self, key):
        if not self.exists(key):
            return False
        shutil.rmtree(self.__get_filename(key))
        return True

    def load(self, key):
        if not self.exists(key):
            return None
        return self.__read_obj(self.__get_filename(key))

    def save(self, key, obj):
        self.__write_obj(self.__get_filename(key), obj)

    def exists(self, key):
        pass

