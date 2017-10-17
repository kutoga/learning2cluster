import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Activation, Concatenate, Lambda, Flatten, Reshape, merge

import tensorflow as tf

def create_weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_pred, y_true):

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return weighted_b_ce

    return weighted_binary_crossentropy

input_shape = (4,)
y_cmp = np.asarray([1, 0, 0, 0])

nw_input0 = Input(shape=input_shape)
nw_input1 = Input(shape=input_shape)

def merge_func(x, y):
    # return K.binary_crossentropy(x, y)
    return create_weighted_binary_crossentropy(1, 0.5)(x, y)

nw = merge([nw_input0, nw_input1], mode=lambda x: merge_func(x[0], x[1]), output_shape=lambda x: x[0])

# nw = K.binary_crossentropy(nw_input0, nw_input1)
# # nw = Activation('linear')(nw)
# nw = tf.convert_to_tensor(nw)
nw = Activation('linear')(nw)

model = Model([nw_input0, nw_input1], [nw, Lambda(lambda x: K.mean(x))(nw)])
model.summary()

y = model.predict([
    np.asarray([[1, 0, 0, 0]]), # y_pred
    np.asarray([[0, 1, 0, 0]])  # y_true
])

# Weights = 1, 1 => [  1.59423847e+01   1.61180954e+01   1.00000015e-07   1.00000015e-07]
print(y) # Print cross entropy and mean

print("done:)")