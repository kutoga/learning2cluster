import numpy as np
from random import Random

from keras.layers import Dense, Input, BatchNormalization, Activation, Reshape
from keras.models import Model

import keras.backend as K

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from core.nn.helper import create_crps_loss

# Create a dummy model that predicts the sum of two numbers from 0-9
n_max = 9
n_inputs = list(range(n_max + 1))
n_outputs = list(range(max(n_inputs) * 2 + 1))

nw_dense_layer_units = [32, 32]

nw_input = Input((2,))
nw = nw_input
for units in nw_dense_layer_units:
    nw = Dense(units)(nw)
    nw = Activation('relu')(nw)
    nw = BatchNormalization()(nw)
nw = Dense(len(n_outputs), activation='softmax')(nw)

model = Model(nw_input, nw)

def my_loss(y_true, y_pred):
    # return K.categorical_crossentropy(y_true, y_pred)

    # cy_true = K.cumsum(y_true, axis=1)
    # cy_pred = K.cumsum(y_pred, axis=1)

    # # my loss
    # si = np.arange(0, int(y_pred.shape[1]))
    # cy_true = K.sum(y_true * si, axis=1)
    # cy_pred = K.sum(y_pred * si, axis=1)
    # d_loss = K.square(cy_true - cy_pred)
    # cce = K.categorical_crossentropy(y_true, y_pred)
    # loss = d_loss + cce

    # my loss #2
    si = np.arange(1, int(y_pred.shape[1] + 1))
    Eyt = K.sum(si * y_true, axis=1)
    yd = y_true - y_pred
    yd2 = K.sqrt(yd)

    Eyt = Reshape((int(y_pred.shape[1]),))(K.repeat(Reshape((1,))(Eyt), int(y_pred.shape[1])))

    loss = K.sum(K.square(yd2 * si - Eyt), axis=1)

    # loss = K.abs(K.sigmoid(cy_true) - K.sigmoid(cy_pred))
    # loss = K.binary_crossentropy(cy_true, cy_pred)

    return loss

model.compile(
    optimizer='SGD',

    # Switch between categorical crossentropy and the CRPS loss

    # loss='categorical_crossentropy',
    # loss=create_crps_loss(),
    # loss=create_crps_loss(use_binary_crossentropy=True),
    loss=my_loss
)

rand = Random(1337)
equally_distributed=True
def generate_data(n):
    x = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n, len(n_outputs)), dtype=np.float32)
    for i in range(n):

        if equally_distributed:

            # We want an equal distributed output: Therefore choose a random result
            s = rand.choice(n_outputs)

            # Choose now two summands
            s1 = rand.choice(list(filter(lambda n: n <= s, n_inputs)))
            s2 = s - s1

            a = s1
            b = s2

        else:

            # The output classes wont be equally distributed (but this should work too!)
            a, b = rand.choice(n_inputs), rand.choice(n_inputs)

        # # Be dirty (make the distributions of the outputs equalliy)
        # a = 0
        # b = rand.choice(n_outputs)

        x[i, 0] = a
        x[i, 1] = b
        y[i, a + b] = 1.
    return x, y

# Train the network for some iterations
minibatch_size = 1000
for itr in range(1000):
    print("Iteration {}".format(itr))
    x, y = generate_data(minibatch_size)
    model.fit(x, y, batch_size=minibatch_size)

# Bar plot
def bar_plot(y_p, title):
    fig, ax = plt.subplots()

    x = n_outputs
    y = y_p.tolist()

    ax.bar(x, y, 0.9, color="blue")
    plt.title(title)

    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show(block=True)

    plt.clf()
    plt.close()

# Do some tests
n_tests = 100
for i in range(n_tests):
    x, y = generate_data(1)
    y_p = model.predict(x)
    bar_plot(y_p[0], "{}+{}=".format(x[0, 0], x[0, 1]))






