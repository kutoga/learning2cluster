import _pickle as pickle
from os import path
from shutil import move

import numpy as np

from keras.layers import Lambda, Activation, Concatenate, GaussianNoise, Dense, Reshape
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


def gaussian_random_layer(shape=(10,), name=None, stddev=1.):
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
    layers.append(Reshape(name=get_name("_RESHAPE0"), shape=shape))
    layers.append(GaussianNoise(name=get_name("_GAUSSIAN0"), stddev=stddev))

    def res(val):
        for layer in layers:
            val = layer(val)
        return val

    return res


def create_weighted_binary_crossentropy(zero_weight, one_weight):

    # # Original function (see tensorflow_backend.py)
    # import tensorflow as tf
    # def _to_tensor(x, dtype):
    #     """Convert the input `x` to a tensor of type `dtype`.
    #
    #     # Arguments
    #         x: An object to be converted (numpy array, list, tensors).
    #         dtype: The destination type.
    #
    #     # Returns
    #         A tensor.
    #     """
    #     x = tf.convert_to_tensor(x)
    #     if x.dtype != dtype:
    #         x = tf.cast(x, dtype)
    #     return x
    # def weighted_binary_crossentropy(target, output, from_logits=False):
    #     """Binary crossentropy between an output tensor and a target tensor.
    #
    #     # Arguments
    #         target: A tensor with the same shape as `output`.
    #         output: A tensor.
    #         from_logits: Whether `output` is expected to be a logits tensor.
    #             By default, we consider that `output`
    #             encodes a probability distribution.
    #
    #     # Returns
    #         A tensor.
    #     """
    #     # Note: tf.nn.sigmoid_cross_entropy_with_logits
    #     # expects logits, Keras expects probabilities.
    #     if not from_logits:
    #         # transform back to logits
    #         _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
    #         output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    #         output = tf.log(output / (1 - output))
    #
    #     return tf.nn.weighted_cross_entropy_with_logits(targets=target,
    #                                                     logits=output,
    #                                                     pos_weight=[zero_weight, one_weight])

    def weighted_binary_crossentropy(y_pred, y_true):

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

