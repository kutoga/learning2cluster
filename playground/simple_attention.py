from keras.models import Model
from keras.layers import Input, Dense, LSTM, Activation, Bidirectional, Concatenate, Reshape, RepeatVector, Lambda, add, multiply

import keras.backend as K

t = 5
nw_input = Input((t, 2))
nw = nw_input
nw = LSTM(3, return_sequences=True)(nw)

l = nw._keras_shape[1]
f = nw._keras_shape[2]
d0 = Dense(4, activation='relu')
d1 = Dense(1, activation='tanh')
new_nw = []
for i in range(l):

    # Calculate the weights
    xi = nw[:, i]
    xi = Lambda(lambda x, xi=xi: xi)(nw)

    weights = []
    for j in range(l):
        xj = nw[:, j]
        xj = Lambda(lambda x, xj=xj: xj)(nw)

        # Calculate the weight
        xij = Concatenate(axis=1)([xi, xj])
        xij = d0(xij)
        xij = d1(xij)
        weights.append(xij)
    weights = Concatenate(axis=1)(weights)
    weights = Activation('softmax')(weights)

    # And now the weighted sum
    weights = Concatenate(axis=2)([Reshape((l, 1))(weights)]*f)
    weighted_output = multiply([nw, weights])
    xi_new = Lambda(lambda x: K.sum(x, axis=1))(weighted_output)
    xi_new = Reshape((1, f))(xi_new)
    new_nw.append(xi_new)
new_nw = Concatenate(axis=1)(new_nw)
nw = new_nw

nw = LSTM(1)(nw)
nw = Dense(1, activation='sigmoid')(nw)

model = Model(nw_input, nw)
model.compile('SGD', 'binary_crossentropy')

print('ja')

