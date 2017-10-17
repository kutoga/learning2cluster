from keras.layers import Dense, Activation, Input
from keras.losses import binary_crossentropy
from keras.models import Model
import keras.backend as K

import numpy as np

input_layer = Input((1,))
nw = input_layer
nw = Dense(32, activation='relu', trainable=False)(nw)
nw = Dense(1, activation='sigmoid', trainable=False)(nw)
factor = nw

loss = binary_crossentropy

# Use now the dirty loss hack:
def my_loss(y_true, y_pred):
    return K.exp(loss(y_true, y_pred)) * (factor + .1)

model = Model(input_layer, nw)
model.compile(optimizer='SGD', loss=my_loss)

print(model.predict(np.asarray([1])))
ones = np.asarray([1, 1])
model.fit(
    ones, ones
)
