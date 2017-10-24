from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Reshape, TimeDistributed, Input

import numpy as np

# <my_nonlinearity>
def build_nonlinearity():

    # Build the nonlinearity
    # EIDT: Mit oder ohne Batch-Norm? Wie viele Units? Mal rumspielen;)
    nl = Sequential()
    nl.add(Dense(4, activation='relu', name='nl_d0', input_shape=(1,)))
    nl.add(BatchNormalization(name='nl_b0'))
    nl.add(Dense(4, activation='relu', name='nl_d1'))
    nl.add(BatchNormalization(name='nl_b1'))
    nl.add(Dense(1))

    # Build a call wrapper (this includes a reshaper etc.)
    def nl_call(input_layer):
        input_shape = input_layer._keras_shape
        tmp_shape = (np.prod(input_shape[1:]), 1)

        # Create the required layers
        nw = input_layer
        nw = Reshape(tmp_shape)(nw)
        nw = TimeDistributed(nl)(nw)
        nw = Reshape(input_shape[1:])(nw)

        return nw

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
from keras import backend as K

batch_size = 512
num_classes = 10
epochs = 1

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
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

activation = Activation('relu')
activation = my_nl

nw_input = Input(input_shape)
nw = nw_input
nw = Conv2D(32, kernel_size=(3, 3),
                 activation='linear',
                 input_shape=input_shape)(nw)
nw = activation(nw)
nw = Conv2D(64, (3, 3), activation='linear')(nw)
nw = activation(nw)
nw = MaxPooling2D(pool_size=(2, 2))(nw)
nw = Dropout(0.25)(nw)
nw = Flatten()(nw)
nw = Dense(128)(nw)
nw = Dropout(0.5)(nw)
nw = Dense(num_classes, activation='softmax')(nw)

model = Model(nw_input, nw)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
# x_train, y_train = unison_shuffled_copies(x_train, y_train)
# x_train = x_train[:1000]
# y_train = y_train[:1000]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

K.set_learning_phase(1) #set learning phase
nw_input = Input((1,))
nw = nw_input
nw = my_nl(nw)
model_nl = Model(nw_input, nw)
x = np.arange(-10, 10, 0.01, dtype=np.float32)
y = model_nl.predict(x)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.savefig('trainable_activation.png')
