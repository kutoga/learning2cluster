import _pickle as pickle
from os import path
from shutil import move
from inspect import getframeinfo, stack

import numpy as np
import scipy.stats as st

from contextlib import contextmanager

from keras.layers import Lambda, Activation, Concatenate, GaussianNoise, Dense, Reshape, Layer, RepeatVector
from keras.models import Sequential
import keras.backend as K

from core.nn.history import History


class AlignedTextTable:
    def __init__(self, add_initial_row=True):
        self.__rows = []
        self.__current_row = None
        if add_initial_row:
            self.new_row()

    def __get_column_length(self, index):
        max_len = -1
        for row in self.__rows:
            if len(row) > index:
                max_len = max(max_len, len(row[index]))
        if max_len < 0:
            max_len = None
        return max_len

    def new_row(self):
        self.__current_row = []
        self.__rows.append(self.__current_row)

    def get_current_cell_count(self):
        if self.__current_row is None:
            return None
        return len(self.__current_row)

    def add_cell(self, cell_content):
        if not isinstance(cell_content, str):
            cell_content = str(cell_content)
        self.__current_row.append(cell_content)

    def get_lines(self):
        # TODO: Make get_column_length ore efficient, currently this (slow) method is executed for each cell, but it
        # could be improved that it is only executed for each column
        lines = ['' for r in range(len(self.__rows))]
        c_i = 0
        while True:
            column_width = self.__get_column_length(c_i)
            if column_width is None:
                break

            for r_i in range(len(self.__rows)):
                if len(self.__rows[r_i]) > c_i:
                    if c_i > 0:
                        lines[r_i] += ' '
                    lines[r_i] += self.__rows[r_i][c_i].ljust(column_width)

            c_i += 1
        return lines

    def print_str(self, f_print=print):
        for line in self.get_lines():
            f_print(line)


class FileWriterHelper:
    def __init__(self, filepath):
        self._filepath = filepath
        self._tmp_filepath = self.__generate_tmp_filepath(filepath)
        self._fh = None

    @property
    def tmp_filepath(self):
        return self._tmp_filepath

    def __generate_tmp_filepath(self, filepath):
        i = 0
        file_dir = path.dirname(filepath)
        filename = path.basename(filepath)
        tmp_filepath = None
        while True:
            tmp_filepath = path.join(file_dir, '.{}.{}.tmp'.format(filename, i))
            if not path.exists(tmp_filepath):
                break
            i += 1
        return tmp_filepath

    def open(self, mode='wb'):
        self.close()
        self._fh = open(self._tmp_filepath, mode)
        return self._fh

    def close(self):
        if self._fh is None:
            return

        # Close the file
        self._fh.close()

        # Move it to the original destination
        move(self._tmp_filepath, self._filepath)

        # Cleanup
        self._fh = None


def __layer_has_weights(l):
    return l.get_weights() is not None and len(l.get_weights()) > 0


def filter_None(x, y):
    x_new = []
    y_new = []
    for i in range(len(y)):
        if y[i] is not None:
            x_new.append(x[i])
            y_new.append(y[i])
    return x_new, y_new


def slice_layer(layer, index, name=None):
    # See: https://github.com/fchollet/keras/issues/890
    # print("Input shape: {} -> {}".format(layer._keras_shape, K.ndim(layer)))
    # print("Output shape: {}".format((1,) + layer._keras_shape[2:]))

    # shape = layer._keras_shape if hasattr(layer, '_keras_shape') else \
    #     tuple(map(lambda x: None if str(x) == '?' else int(str(x)), layer.shape.dims))
    shape = layer._keras_shape

    res = Lambda(
        lambda x: x[:, index:(index + 1), :],
        output_shape=(1,) + shape[2:],
        name=name,
        trainable=False
    )(layer)
    # print("Output shape: {} -> {}".format(res._keras_shape, K.ndim(res)))
    return res


def concat_layer(axis=-1, name=None, input_count=None):
    if input_count is None or input_count > 1:
        return Concatenate(axis=axis, name=name)
    return Activation('linear', name=name)


class DynamicGaussianNoise(Layer):

    def __init__(self, stddev=1., mean=0., only_execute_for_training=True, **kwargs):
        super(DynamicGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.mean = mean
        self.only_execute_for_training = only_execute_for_training

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=self.mean,
                                            stddev=self.stddev)
        if self.only_execute_for_training:
            return K.in_train_phase(noised, inputs, training=training)
        else:
            return noised()

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gaussian_random_layer(shape=(10,), name=None, stddev=1., mean=0., only_execute_for_training=True):
    """
    Just generate a layer with random numbers. Unfortunately this layer has to be called with an input tensor, but the
    values of this input tensor are not used at all. That's ugly, but currently this cannot be avoided.
    """

    # TODO: Fix the return value of this function in a way that it is possible to use save_weights etc. Do this as soon as this layer is used

    def get_name(suffix):
        if name is None:
            return None
        return "{}_{}".format(name, suffix)

    layers = []
    layers.append(Dense(name=get_name("_DENSE0"), units=np.prod(shape), kernel_initializer='zeros', bias_initializer='zeros', trainable=False))
    layers.append(Reshape(name=get_name("_RESHAPE0"), target_shape=shape))
    layers.append(DynamicGaussianNoise(name=get_name("_GAUSSIAN0"), stddev=stddev, mean=mean, only_execute_for_training=only_execute_for_training))

    def res(val):
        for layer in layers:
            val = layer(val)
        return val

    return res


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def create_crps_loss(use_binary_crossentropy=False, summation_f=K.square):
    """
    TODO: Add paper

    :param use_binary_crossentropy: Use the binary cross-entropy for the summantion
    :param summation_f: A summation function for the difference between the cumulative distribution. It is recommended that f(x)=f(-x) and f(0)=0
    :return:
    """
    def crps_loss(y_true, y_pred):
        if use_binary_crossentropy:
            d = K.binary_crossentropy(
                K.cumsum(y_true),
                K.cumsum(y_pred)
            )
        else:
            d = summation_f(
                K.cumsum(y_pred - y_true)
            ) / y_true.shape[1]
        return K.sum(d)
    return crps_loss


def get_caller():
    # See: https://stackoverflow.com/a/24439444/916672
    return getframeinfo(stack()[2][0])


@contextmanager
def np_show_complete_array():
    # See: https://stackoverflow.com/a/45831462/916672
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**oldoptions)


def mean_confidence_interval(data, confidence=0.95):
    # See: https://stackoverflow.com/a/34474255/916672
    return st.t.interval(confidence, len(data)-1, loc=np.mean(data), scale=st.sem(data))


def linear_inerpolation_for_None_values(values):
    """
    :param values:
    :return: False if nothing could be done (values only contains None values), otherwise true
    """

    # Replace all Nones at the beginning
    for i in range(len(values)):
        if values[i] is not None:
            if i > 0:
                values[0:i] = [values[i]] * i
            break
        if i == len(values) - 1:

            # There are only None-values in the given list: Nothing can be done, return None
            return False

    # Replace all Nones at the end
    # for i in reversed(range(len(values))):
    for i in range(len(values) - 1, -1, -1):
        if values[i] is not None:
            if i < len(values) - 1:
                values[(i + 1):len(values)] = [values[i]] * (len(values) - 1 - i)
            break

    # Find for each array position the previous and the next non-None value
    previous_value_i = [None] * len(values)
    next_value_i = [None] * len(values)
    tmp = 0
    for config in [(previous_value_i, range(len(values))), (next_value_i, range(len(values) - 1, -1, -1))]:
        target_list, indices = config
        for i in indices:
            if values[i] is None:
                target_list[i] = tmp
            else:
                target_list[i] = i
                tmp = i

    # Now the averaging process is no longer that hard: Do a linear interpolation
    for i in range(len(values)):
        if values[i] is None:
            i_prev = previous_value_i[i]
            i_next = next_value_i[i]
            d_v = values[i_next] - values[i_prev]

            values[i] = values[i_prev] + (i - i_prev) / (i_next - i_prev) * d_v

    # Everything is done:)
    return True


def sliding_window_average_for_notNone(values, window_range=2):
    """
    window_length = 2*window_range+1
    :param values:
    :param window_range:
    :return:
    """
    input_len = len(values)

    # If the input is empty: Return an empty array
    if input_len == 0:
        return []

    # If the input shorter than or euqal to 1+window_range this is a real special case: Then we get a constant output
    if input_len < (1 + window_range):
        return [sum(values) / input_len] * input_len

    # Prepare the output array and also some running values
    result = [0.] * len(values)
    current_sum = 0.
    current_divisor = 0

    # Trivial implementation (maybe a bit slow: It could be faster if the first and the last window_range values are handled seperatly, then no "ifs" would be required)
    current_sum += sum(values[0:window_range])
    current_divisor += window_range
    for i in range(len(values)):
        i_remove = i - window_range - 1
        i_add = i + window_range
        if i_remove >= 0:
            current_sum -= values[i_remove]
            current_divisor -= 1
        if i_add < input_len:
            current_sum += values[i_add]
            current_divisor += 1
        result[i] = current_sum / current_divisor

    return result


def sliding_window_average(values, window_range=2, interpolation_for_None='linear'):
    """
    window_length = 2*window_range+1
    :param values:
    :param window_range:
    :return:
    """
    values = list(values)  # Copy the array (this prevents that it is modified)

    # Replace the None values if there are any. First test the array for None values, because the interpolation may be
    # very expensive
    if any(map(lambda v: v is None, values)):
        if interpolation_for_None == 'linear':
            interpolation_ok = linear_inerpolation_for_None_values(values)
        else:
            raise ValueError()
        if not interpolation_ok:

            # The interpolation doesn't work: Just return the input array (probably it only contains None values)
            return values

    # # Do the averaging
    # def get_value(i):
    #     i_start = max(0, i - window_range)
    #     i_end = min(len(values), i + 1 + window_range)
    #     return sum(values[i_start:i_end]) / (i_end - i_start)
    # return list(map(get_value, range(len(values))))

    # Do the averaging a bit faster
    return sliding_window_average_for_notNone(values, window_range)

    # Maybe pandas.rolling_average would be event faster? Unfortunately its handling for the edge values sucks. But sliding_window_average_for_notNone
    # and pandas.rolling_average scale linear (rolling_average is about 6 times faster, also sliding_window_average_for_notNone requires only
    # 1.2s on my notebook for 1'000'000 values; thats ok)

__MODEL_FILE_WEIGHTS_SUFFIX = '.weights.pkl'
__MODEL_FILE_HISTORY_SUFFIX = '.history.pkl'
__MODEL_FILE_OPTIMIZER_SUFFIX = '.optimizer.pkl'


def __extract_layers_wight_weights(layers):
    return filter(__layer_has_weights, layers)


def save_optimizer_state(model, base_filename):
    optimizer = model.optimizer

    config = optimizer.get_config()
    weights = optimizer.get_weights()
    optimizer_type = type(optimizer)
    state = {
        'config': config,
        'weights': weights,
        'type': {
            'module': optimizer_type.__module__,
            'name': optimizer_type.__name__
        }
    }

    filename = base_filename + __MODEL_FILE_OPTIMIZER_SUFFIX
    print('Save optimizer state to {}...'.format(filename))
    fw_helper = FileWriterHelper(filename)
    with fw_helper.open('wb') as state_file:
        pickle.dump(state, state_file)
        fw_helper.close()


def load_optimizer_state(model, base_filename):
    filename = base_filename + __MODEL_FILE_OPTIMIZER_SUFFIX
    if isinstance(model, Sequential):
        model = model.model

    print('Load optimizer state from {}...'.format(filename))
    with open(filename, 'rb') as state_file:
        state = pickle.load(state_file)
        state_file.close()

    # Load the optimizer
    # TODO: Do not use exec / eval
    optimizer_type = state['type']
    exec("from {} import {}".format(optimizer_type['module'], optimizer_type['name']))
    optimizer = eval(optimizer_type['name']).from_config(state['config']) # TODO: Users could do evil things here, currently this is ok, because i am the only user
    model.optimizer = optimizer

    # TODO: This is ugly and i dont know why it has to be done (but it has to). See: https://github.com/fchollet/keras/blob/master/keras/models.py
    if model.train_function is not None:
        print("WARNING: Removeingh the train function (required for loading the optimizer state)")
        model.train_function = None
    model._make_train_function()

    # Load the weights
    model.optimizer.set_weights(state['weights'])


def save_weights(model, base_filename):
    layers_with_weights = __extract_layers_wight_weights(model.layers)
    weights = {
        layer.name: layer.get_weights() for layer in layers_with_weights
    }
    filename = base_filename + __MODEL_FILE_WEIGHTS_SUFFIX
    print('Save weights to {}...'.format(filename))
    fw_helper = FileWriterHelper(filename)
    with fw_helper.open('wb') as weights_file:
        pickle.dump(weights, weights_file)
        fw_helper.close()


def load_weights(model, base_filename, print_unitialized_target_layers=True):
    filename = base_filename + __MODEL_FILE_WEIGHTS_SUFFIX
    print('Load weights from {}...'.format(filename))
    with open(filename, 'rb') as weights_file:
        weights = pickle.load(weights_file)
        weights_file.close()
    layers_with_weights = set(__extract_layers_wight_weights(model.layers))
    initialized_layers = set()
    for layer in list(layers_with_weights):
        layer_name = layer.name
        if layer_name in weights:
            layer.set_weights(weights[layer_name])
            layers_with_weights.remove(layer)
            initialized_layers.add(layer_name)

    # If required: Print all non-initialized layers
    if print_unitialized_target_layers:
        if len(layers_with_weights) == 0:
            print("All layers with weights are initialized")
        else:
            print("Non-initialized layers with weights:")
            for layer in layers_with_weights:
                print("- {} ({})".format(layer.name, layer))

    # Return the initialized layers and also the not initialized layers with weights. In general we do not care about
    # uninitialized layers without weights.
    return initialized_layers, layers_with_weights


def save_history(history, base_filename):
    filename = base_filename + __MODEL_FILE_HISTORY_SUFFIX
    print('Save history to {}...'.format(filename))
    fw_helper = FileWriterHelper(filename)
    with fw_helper.open('wb') as history_file:
        pickle.dump(history, history_file)
        fw_helper.close()


def load_history(base_filename):
    filename = base_filename + __MODEL_FILE_HISTORY_SUFFIX
    print('Load history from {}...'.format(filename))
    with open(filename, 'rb') as history_file:
        try:
            history = pickle.load(history_file)
        except:
            history = History()
        history_file.close()
    return history

#
# #
# # Notes for load and save model functions:
# # Keras default functions may be buggy (see the "quick&dirty" code which was created before this project). Therefore
# # some dirty and hacky tricks have to be done. Maybe futute Keras versions are not as buggy and the Keras functions
# # directly may be used.
# #
# __MODEL_FILE_WEIGHTS_SUFFIX = '.weights.bin'
# __MODEL_FILE_JSON_MODEL_SUFFIX = '.model.json'
# __MODEL_FILE_YAML_MODEL_SUFFIX = '.model.yaml'
# __MODEL_FILE_CONFIG_MODEL_SUFFIX = '.model.config'
#
# # The binary model is currently only used to store the model. Loading fails for some models. Maybe future versions
# # of Keras no longer have these bugs and the binary model may be used. It is nice, because it contains the model and
# # also all weights.
# __MODEL_FILE_MODEL_SUFFIX = '.model.bin'
#
#
# def save_weights_old(model, base_filename):
#     model.save(base_filename + __MODEL_FILE_MODEL_SUFFIX)
#     model.save_weights(base_filename + __MODEL_FILE_WEIGHTS_SUFFIX)
#     with open(base_filename + __MODEL_FILE_JSON_MODEL_SUFFIX, 'w') as json_model:
#         json_model.write(model.to_json())
#         json_model.close()
#     with open(base_filename + __MODEL_FILE_CONFIG_MODEL_SUFFIX, 'wb') as config_model:
#         pickle.dump(model.get_config(), config_model)
#         config_model.close()
#     with open(base_filename + __MODEL_FILE_YAML_MODEL_SUFFIX, 'w') as yaml_model:
#         yaml_model.write(model.to_yaml())
#         yaml_model.close()
#
#
# def load_weights_old(model, base_filename):
#     with open(base_filename + __MODEL_FILE_JSON_MODEL_SUFFIX) as model_file:
#         json_model = model_file.read()
#         model_file.close()
#     with open(base_filename + __MODEL_FILE_CONFIG_MODEL_SUFFIX, 'rb') as model_file:
#         config_model = pickle.load(model_file)
#         model_file.close()
#     with open(base_filename + __MODEL_FILE_CONFIG_MODEL_SUFFIX, 'rt') as model_file:
#         yaml_model = model_file.read()
#         model_file.close()
#     # loaded_model = model_from_json(json_model)
#     # loaded_model = model_from_config(config_model)
#     loaded_model = model_from_yaml(yaml_model)
#     loaded_model.load_weights(base_filename + __MODEL_FILE_WEIGHTS_SUFFIX)
#     copy_weights(loaded_model, model)
#
#
# def copy_weights(source_model, target_model, target_prefix=None, print_unitialized_target_layers=True):
#
#     target_layers_with_weights = set(__extract_layers_wight_weights(target_model.layers))
#     initialized_layers = set()
#
#     # Create a dictionary with all target layers
#     target_layers = {}
#     for layer in filter(lambda l: l.name is not None, target_model.layers):
#         target_layers[layer.name] = layer
#
#     # Copy all weights for all available layers
#     for layer in filter(lambda l: l.name is not None, source_model.layers):
#         if layer.name is not None and __layer_has_weights(layer):
#             target_name = ('' if target_prefix is None else target_prefix) + layer.name
#             if target_name in target_layers:
#                 weight_count = sum(np.product(w.shape) for w in layer.get_weights())
#                 print('Copy {} weights from layer "{}" to layer "{}"...'.format(weight_count, layer.name, target_name))
#                 try:
#                     target_layer = target_layers[target_name]
#                     target_layers[target_name].set_weights(layer.get_weights())
#                     target_layers_with_weights.remove(target_layer)
#                     initialized_layers.add(target_layer)
#                 except:
#                     print('Failed. Source shape: {}, Target shape: {}'.format(
#                         np.asarray(layer.get_weights()).shape,
#                         np.asarray(target_layers[target_name].get_weights()).shape
#                     ))
#
#     # If required: Print all uinitialized layers
#     if print_unitialized_target_layers:
#         if len(target_layers_with_weights) == 0:
#             print("All layers with weights are initialized")
#         else:
#             print("Unititialized layers with weights:")
#             for layer in target_layers_with_weights:
#                 print("- {} ({})".format(layer.name, layer))
#
#     # Return the initialized layers and also the not initialized layers with weights. In general we do not care about
#     # uninitialized layers without weights.
#     return initialized_layers, target_layers_with_weights
#


def get_val_at(input_layer, i_i):
    def at(val, indices):
        for index in indices:
            val = val[index]
        return val
    return Lambda(lambda x: at(x, i_i), output_shape=(1,))(input_layer)


def get_cluster_centers(embeddings, cluster_classification):
    """
    :param embeddings: A list of embeddings
    :param cluster_classification: A list of softmaxs
    :param base_name:
    :return:
    """
    cluster_count = int(str(cluster_classification[0].shape[2]))
    embedding_dim = int(str(embeddings[0].shape[2]))
    s = [0] * cluster_count
    c = [1e-10] * cluster_count
    for e_i in range(len(embeddings)):
        embedding = embeddings[e_i]
        current_cluster_classification = cluster_classification[e_i]
        current_cluster_classification = Reshape((cluster_count, 1))(current_cluster_classification)
        for c_i in range(cluster_count):
            p = Reshape((1,))(slice_layer(current_cluster_classification, c_i)) #get_val_at(current_cluster_classification, [1, c_i])
            p = RepeatVector(embedding_dim)(p)
            s[c_i] = Lambda(lambda x: x * embedding + s[c_i])(p)
            c[c_i] = Lambda(lambda x: x + c[c_i])(p)
    for c_i in range(cluster_count):
        s[c_i] = Lambda(lambda x: x / c[c_i])(s[c_i])
        s[c_i] = slice_layer(s[c_i], 0)
    return s


def get_cluster_separation(cluster_centers, cluster_classification, distance_f=lambda x, y: K.sum(K.square(x - y), axis=2)):
    """
    Important: The result has to be negated if it is used inside a loss function.

    :param cluster_centers: A list of [1, N] cluster centers
    :param distance_f: A distance function. The current implementation assumes that d(x, x) = 0 and d(x, y) = d(y, x)
    :param base_name:
    :return:
    """
    distance_sum = 0
    counter = 0
    mcluster_classification = Concatenate(axis=1)(cluster_classification) # (E, 7) -> transpose -> (7, E) -> sum(axis=2)
    weights = Lambda(lambda x: K.sum(x, axis=1))(mcluster_classification)
    weights = Reshape((len(cluster_centers), 1))(weights)
    for c_i in range(len(cluster_centers)):
        current_distance_sum = 0
        for c_j in range(len(cluster_centers)):
            c_source = cluster_centers[c_i]
            c_target = cluster_centers[c_j]
            c_distance = Lambda(lambda c_source: distance_f(c_source, c_target))(c_source)
            current_distance_sum = Lambda(lambda c_distance: current_distance_sum + c_distance)(c_distance)
            counter += 1

        # Calculate w_i
        w_i = Reshape((1,))(slice_layer(weights, c_i))
        distance_sum = Lambda(lambda current_distance_sum: current_distance_sum * w_i + distance_sum)(current_distance_sum)

    avg_distance = Lambda(lambda distance_sum: distance_sum / (len(cluster_classification) * len(cluster_centers)))(distance_sum)
    return avg_distance


def get_cluster_cohesion(cluster_centers, embeddings, cluster_classification, distance_f=lambda x, y: K.sum(K.square(x - y), axis=2)):
    """
    Calculate cluster cohesion and return the resulting layer.
    :param cluster_centers: A list of [1, N] cluster centers
    :param embeddings: A list of embeddings / [1, N]
    :param cluster_classification:
    :param distance_f:
    :return:
    """
    cluster_count = int(str(cluster_classification[0].shape[2]))
    distance_sum = 0

    for e_i in range(len(embeddings)):
        embedding = embeddings[e_i]
        current_cluster_classification = cluster_classification[e_i]
        current_cluster_classification = Reshape((cluster_count, 1))(current_cluster_classification)
        for c_i in range(len(cluster_centers)):
            cluster_center = cluster_centers[c_i]
            p = Reshape((1,))(slice_layer(current_cluster_classification, c_i)) #get_val_at(current_cluster_classification, [1, c_i])
            distance = Lambda(lambda cluster_center: distance_f(cluster_center, embedding))(cluster_center)
            distance = Lambda(lambda distance: distance * p)(distance)
            distance_sum = Lambda(lambda distance: distance_sum + distance)(distance)

    avg_distance = Lambda(lambda distance_sum: distance_sum / len(embeddings))(distance_sum)
    return avg_distance


if __name__ == '__main__':
    from random import random
    from time import time
    import pandas as pd

    count = 1000000
    window_range = count // 10
    window_len = 2 * window_range + 1

    y = [random() for i in range(count)]

    print("Start")
    t_start = time()
    y1 = sliding_window_average_for_notNone(y, window_range)
    t_end = time()
    print("Required time [s]: {}".format(t_end - t_start))

    print("Start")
    t_start = time()
    y1 = list(pd.rolling_mean(np.asarray(y), window_len, center=True))
    t_end = time()
    print("Required time [s]: {}".format(t_end - t_start))

    print("Start")
    t_start = time()
    def get_value(i):
        i_start = max(0, i - window_range)
        i_end = min(len(y), i + 1 + window_range)
        return sum(y[i_start:i_end]) / (i_end - i_start)
    y2 = list(map(get_value, range(len(y))))
    t_end = time()
    print("Required time [s]: {}".format(t_end - t_start))

    print("Start")
    t_start = time()
    l = len(y)
    def get_value(i):
        i_start = max(0, i - window_range)
        i_end = min(l, i + 1 + window_range)
        return sum(y[i_start:i_end]) / (i_end - i_start)
    y3 = list(map(get_value, range(l)))
    t_end = time()
    print("Required time [s]: {}".format(t_end - t_start))


