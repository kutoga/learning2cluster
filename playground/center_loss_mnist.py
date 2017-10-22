import functools

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, Flatten, Input

import keras.backend as K

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# This code is based on:
# https://github.com/fchollet/keras/issues/6929
update_ops = []
def _center_loss_func(features, labels, alpha, num_classes,
                      centers, feature_dim):
    assert feature_dim == features.get_shape()[1]
    labels = K.reshape(labels, [-1])
    labels = tf.to_int32(labels)
    centers_batch = tf.gather(centers['c'], labels)
    diff = (1 - alpha) * (centers_batch - features)

    # Should this be a reference? Because the current value is not used
    centers['c'] = tf.scatter_sub(centers['c'], labels, diff)
    update_ops.append(centers['c'])

    loss = tf.reduce_mean(K.square(features - centers_batch))
    return loss

def get_center_loss(alpha, num_classes, feature_dim, p_lambda=1.):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # Each output layer use one independed center: scope/centers
    centers = {'c': K.zeros([num_classes, feature_dim])}
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return p_lambda * _center_loss_func(y_pred, y_true, alpha,
                                 num_classes, centers, feature_dim)
    return center_loss

# Create a simple network for mnsit. It should create nice clusters
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
y_train_c = keras.utils.to_categorical(y_train, 10)
y_test_c = keras.utils.to_categorical(y_test, 10)

nw_input = Input((28, 28, 1))
nw = nw_input

nw = Conv2D(32, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = MaxPool2D()(nw)

nw = Conv2D(64, (3, 3), padding='same')(nw)
nw = Activation('relu')(nw)
nw = BatchNormalization()(nw)
nw = MaxPool2D()(nw)

nw = Flatten()(nw)
nw = Dense(64, activation='relu')(nw)
nw = BatchNormalization()(nw)
nw = Dense(2, activation='tanh')(nw)
nw = BatchNormalization()(nw)

# Output 1: softmax
nw_sm = Dense(10, activation='softmax', name='sm')(nw)

# # Output 2: Center-Loss
# nw_cl = Dense(2, activation='tanh', name='cl')(nw)
nw_cl = Activation('linear', name='cl')(nw)

model = Model([nw_input], [nw_sm, nw_cl])
model.summary()

# Visualization is always great:D
def visualize(x, y, fname=None):
    y_sm, y_cl = model.predict([x])

    fig, ax = plt.subplots()

    classes = np.unique(y)
    for cls in sorted(classes):
        c_y_cl = y_cl[y == cls]
        px = c_y_cl[:, 0]
        py = c_y_cl[:, 1]
        ax.scatter(px, py, alpha=0.8)

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    if fname is None:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.savefig(fname)
        plt.clf()
        plt.close()

# with tf.control_dependencies(update_ops):
model.compile(
    'adam',
    loss={
        'sm': 'categorical_crossentropy',
        'cl': get_center_loss(.5, 10, 2),
    },
    metrics={
        'sm': 'categorical_accuracy'
    },
    # fetches=update_ops
)

for i in range(100000):
    visualize(x_test, y_test, 'E:\\out_cl_epoch{}.png'.format(i))
    # with tf.control_dependencies(update_ops):
    model.fit([x_train], {
        'sm': y_train_c,
        'cl': y_train
    }, batch_size=320, epochs=1)
    # K.get_session().run(update_ops)

