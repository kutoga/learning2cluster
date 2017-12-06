from keras.models import Model
from keras.layers import Input, Dense, LSTM, Reshape, UpSampling1D, Flatten

import numpy as np

nw_input = Input((1,))
nw = nw_input
nw = Dense(10)(nw)

# Create a time axis
nw = Reshape((5, 2))(nw)

lstm = LSTM(2, return_sequences=True)

nw = lstm(nw)
nw = UpSampling1D(2)(nw)
nw = lstm(nw)

nw = Flatten()(nw)
nw = Dense(1, activation='sigmoid')(nw)

model = Model(nw_input, nw)
model.compile(
    'adam',
    'binary_crossentropy'
)
model.predict(np.zeros((3, 1), dtype=np.float32))

nw_input = Input((1,))
nw = nw_input
nw = Dense(20)(nw)
nw = Reshape((10, 2))(nw)
nw = lstm(nw)
nw = Flatten()(nw)
nw = Dense(1, activation='sigmoid')(nw)
model2 = Model(nw_input, nw)
model2.compile(
    'adam',
    'binary_crossentropy'
)
model2.predict(np.zeros((3, 1), dtype=np.float32))



