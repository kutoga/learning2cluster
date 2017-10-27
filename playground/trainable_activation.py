from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Reshape, TimeDistributed, Input, Dropout, Concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GaussianNoise
from keras.regularizers import l2
from keras.layers.noise import GaussianNoise

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras.backend as K

K.set_learning_phase(1) #set learning phase

regularizer = l2(0.0001)
test_name = 'relu_v04'
base_activation = Activation('relu')
plot_x_min = -2
plot_x_max = 2
mixed = False

# test_name = 'mixed_v01'
# mixed = True

# <my_nonlinearity>
nl_counter = 0
def build_nonlinearity():

    if not mixed:
        # # Build the nonlinearity
        # nl = Sequential()
        #
        # # 3 Relu-layers: It should be possible to approximate quite many functions with these layers
        # nl.add()
        # nl.add(base_activation)
        # nl.add(GaussianNoise(0.01))
        # nl.add(Dense(8, name='nl_d1', kernel_regularizer=regularizer))
        # nl.add(base_activation)
        # # nl.add(Dense(32, name='nl_d2'))
        # # nl.add(base_activation)
        # nl.add(Dense(1))

        nl_input = Input((1,))
        nl = nl_input
        nl = Dense(8, kernel_regularizer=regularizer, input_shape=(1,))(nl)
        nld0 = nl

        nl = base_activation(nl)
        nl = GaussianNoise(0.001)(nl)
        nl = Dense(8, kernel_regularizer=regularizer, input_shape=(1,))(nl)
        nl = base_activation(nl)
        nld1 = add([nld0, nl])
        nl = nld1

        nl = base_activation(nl)
        nl = GaussianNoise(0.001)(nl)
        nl = Dense(8, kernel_regularizer=regularizer, input_shape=(1,))(nl)
        nl = base_activation(nl)
        nld2 = add([nld1, nl])
        nl = nld2

        nl = Dense(1, kernel_regularizer=regularizer)(nl)
        nl = add([nl_input, nl])

        nl = Model(nl_input, nl)


    else:
        nl_input = Input((1,))
        nl = nl_input

        nl0 = nl
        nl0 = Dense(8)(nl0)
        nl0 = LeakyReLU()(nl0)

        nl1 = nl
        nl1 = Dense(8)(nl1)
        nl1 = Activation('tanh')(nl1)

        nl = Concatenate(axis=1)([nl0, nl1])

        nl0 = nl
        nl0 = Dense(8)(nl0)
        nl0 = LeakyReLU()(nl0)

        nl1 = nl
        nl1 = Dense(8)(nl1)
        nl1 = Activation('tanh')(nl1)

        nl = Concatenate(axis=1)([nl0, nl1])
        nl = Dense(1)(nl)

        nl = Model(nl_input, nl)

    # Build a call wrapper (this includes a reshaper etc.)
    def nl_call(input_layer):
        input_shape = input_layer._keras_shape
        tmp_shape = (np.prod(input_shape[1:]), 1)

        # We want to have only a single layer in the summary of the model, so wrap the layers into a own model
        nl_nw_input = Input(input_shape)
        nl_nw = nl_nw_input
        nl_nw = Reshape(tmp_shape)(nl_nw)
        nl_nw = GaussianNoise(0.01)(nl_nw)
        nl_nw = TimeDistributed(nl)(nl_nw)
        nl_nw = Reshape(input_shape[1:])(nl_nw)

        global nl_counter
        res = Model(nl_nw_input, nl_nw, name="NL_{}_{}".format(test_name, nl_counter))(input_layer)
        nl_counter += 1
        return res

    return nl_call

my_nl = build_nonlinearity()
# </my_nonlinearity>


nw_input = Input((2,))
nw = nw_input
nw = Dense(16, input_shape=(2,))(nw)

# model.add(Reshape((16, 1)))
# model.add(TimeDistributed(my_nl))
# model.add(Reshape((16,)))
nw = my_nl(nw)

nw = BatchNormalization()(nw)
nw = Dense(1)(nw)

model = Model(nw_input, nw)

x = np.zeros((2, 2), dtype=np.float32)
y = model.predict(x)


# Create now a real model, e.g. for mnist
# Copied from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 512
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

activation = Activation('relu')
activation = my_nl

nw_input = Input(input_shape)
nw = nw_input
nw = Dropout(0.25)(nw)
nw = Conv2D(32, kernel_size=(3, 3),
                 activation='linear')(nw)
nw = activation(nw)
nw = BatchNormalization()(nw)
nw = Conv2D(64, (3, 3), activation='linear')(nw)
nw = activation(nw)
nw = BatchNormalization()(nw)
nw = MaxPooling2D(pool_size=(2, 2))(nw)
nw = Dropout(0.25)(nw)
nw = Flatten()(nw)
nw = Dense(128)(nw)
nw = activation(nw)
nw = Dropout(0.25)(nw)
nw = Dense(num_classes, activation='softmax')(nw)

model = Model(nw_input, nw)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta()
              #,metrics=['accuracy']
)
model.summary()
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
# x_train, y_train = unison_shuffled_copies(x_train, y_train)
# x_train = x_train[:1000]
# y_train = y_train[:1000]

nw_input = Input((1,))
nw = nw_input
nw = my_nl(nw)
model_nl = Model(nw_input, nw)

def plot_trainable_activation(index):
    x = np.arange(plot_x_min, plot_x_max, (plot_x_max - plot_x_min) / 1000., dtype=np.float32)
    K.set_learning_phase(0)
    y = model_nl.predict(x)
    K.set_learning_phase(1)
    plt.plot(x, y)
    plt.savefig('trainable_activation_{}_{}.png'.format(test_name, index))
    plt.clf()
    plt.close()

plot_trainable_activation(0)
for i in range(epochs):
    c_x_train, c_y_train = unison_shuffled_copies(x_train, y_train)
    c_x_train = c_x_train[:10000]
    c_y_train = c_y_train[:10000]
    validation_data = None
    if i % 6 == 0:
        validation_data = (x_test, y_test)
    model.fit(c_x_train, c_y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=validation_data)
    plot_trainable_activation(i + 1)

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

