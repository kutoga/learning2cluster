import numpy as np
from random import Random

from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.models import Model

import matplotlib.pyplot as plt

from core.nn.helper import create_crps_loss

# Create a dummy model that predicts the sum of two numbers from 0-9
n_max = 9
n_inputs = list(range(n_max + 1))
n_outputs = list(range(max(n_inputs) * 2 + 1))

nw_dense_layer_units = [64, 64, 64]

nw_input = Input((2,))
nw = nw_input
for units in nw_dense_layer_units:
    nw = Dense(units)(nw)
    nw = BatchNormalization()(nw)
    nw = Activation('relu')(nw)
nw = Dense(len(n_outputs), activation='softmax')(nw)

model = Model(nw_input, nw)
model.compile(
    optimizer='SGD',

    # Switch between categorical crossentropy and the CRPS loss

    # loss='categorical_crossentropy',
    loss=create_crps_loss(),
)

rand = Random(1337)
def generate_data(n):
    x = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n, len(n_outputs)), dtype=np.float32)
    for i in range(n):
        a, b = rand.choice(n_inputs), rand.choice(n_inputs)

        # Be dirty (make the distributions of the outputs equalliy)
        a = 0
        b = rand.choice(n_outputs)

        x[i, 0] = a
        x[i, 1] = b
        y[i, a + b] = 1.
    return x, y

# Train the network for some iterations
minibatch_size = 1000
for itr in range(2000):
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
    plt.show(block=True)

    plt.clf()
    plt.close()

# Do some tests
n_tests = 100
for i in range(n_tests):
    x, y = generate_data(1)
    y_p = model.predict(x)
    bar_plot(y_p[0], "{}+{}=".format(x[0, 0], x[0, 1]))






