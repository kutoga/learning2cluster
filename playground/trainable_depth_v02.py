from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.regularizers import l2

class DynamicLayer:
    def __init__(self, name, f_layer, h_step=Activation('relu'), w_init=1.0, w_regularizer=l2(1e-11), layer_reg=l2(1e-2)):
        if not isinstance(f_layer, list):
            f_layer = [f_layer]

        self.f_layer = f_layer
        self.h_step = h_step
        self.w_init = w_init

class TDModel:
    def __init__(self):
        self.layers = []

    def append(self, layer):
        if isinstance(layer, list):
            for l in layer:
                self.append(l)
            return
        self.layers.append(layer)

    def __add__(self, layer):
        self.append(layer)

    def build(self):
        pass

    def train_step(self, x, y, validation_data=None):
        pass


model = TDModel()
model += Input((2,))
model += Dense(12)
model += DynamicLayer('d0', [
    lambda *args: Dense(*args),
    lambda *args: Dense(*args),
    lambda *args: BatchNormalization()
])
model += Dense(1, activation='sigmoid')

model.build()
