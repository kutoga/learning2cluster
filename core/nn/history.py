
class History:
    def __init__(self):
        self.__data = {}

    def __getitem__(self, item):
        if item in self.__data:
            return self.__data[item]
        return [None] * self.length()

    def __len__(self):
        return len(self.__data)

    def __get_index_f(self, key, f_cmp, default=None):
        if key not in self.__data:
            return default
        arr = self.__data[key]
        v = f_cmp(filter(lambda x: x is not None, arr), default=None)
        if v is None:
            return None
        return arr.index(v)

    def get_latest_values(self, key, include_none_values=False, n=1):
        if key not in self.__data:
            return []
        values = []
        for x in reversed(self.__data[key]):
            if include_none_values or (x is not None):
                values.insert(0, x)
                if len(values) >= n:
                    break
        return values

    def keys(self):
        return self.__data.keys()

    def values(self):
        return self.__data.values()

    def get_or_create_item(self, item):
        if item not in self.__data:
            self.__data[item] = self[item]
        return self.__data[item]

    def load_keras_history(self, hist): # load keras or History history objects
        if len(hist) == 0:
            return
        map_length = len(next(iter(hist.values())))
        length = self.length()

        not_available_props = set(self.__data.keys())
        for key in hist.keys():
            if key not in self.__data:
                self.__data[key] = [None] * length
            self.__data[key] += hist[key]
            if key in not_available_props:
                not_available_props.remove(key)

        # Insert None values for the not available properties
        for prop in map(lambda p: self[p], not_available_props):
            prop += [None] * map_length

    def length(self):
        if len(self.__data) == 0:
            return 0
        return len(next(iter(self.__data.values())))

    def get_epoch_indices(self):
        return range(1, self.length() + 1)

    def get_min_index(self, key):
        return self.__get_index_f(key, min)

    def get_max_index(self, key):
        return self.__get_index_f(key, max)
