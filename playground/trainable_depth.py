

# In this file I try to create a neural network that has a trainable depth; for this example the depth is limited to N
# (in general it could be any natural number, but i just like to create a proof-of-concept)

# First we require a function delta(x)=Relu(exp(x-w)) where w is a trainable weight


from keras.layers import Layer, InputSpec
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.backend as K

class DeltaF(Layer):
    def __init__(self, alpha_regularizer=None, default_layer_count=1,
                 **kwargs):
        super(DeltaF, self).__init__(**kwargs)
        self.supports_masking = True
        if default_layer_count < K.epsilon():
            print("The default layer count should never be smaller or equal to 0, otherwise it will never change. It is recommended to use at least 0.5.")
        self.default_layer_count = default_layer_count
        self.alpha_initializer = initializers.Constant(default_layer_count)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)

    def build(self, input_shape):
        param_shape = [1]
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     regularizer=self.alpha_regularizer,
                                     initializer=self.alpha_initializer)
        # Set input spec
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs, mask=None):
        return K.relu(1 - K.exp(inputs - self.alpha))

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'default_layer_count': self.default_layer_count
        }
        base_config = super(DeltaF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.models import Model
from keras.layers import Input
from keras.losses import mean_squared_error
import numpy as np

# Does the DeltaF function layer work? It does:)
if False:
    nw_input = Input((1,))
    df = DeltaF()
    nw = df(nw_input)

    model = Model(nw_input, nw)
    model.compile('SGD', mean_squared_error)

    def my_f(x, w):
        return np.maximum(0, 1 - np.exp(x - w))

    def generate_batch(w, n=1000):
        x = np.random.uniform(-1, w + 1, (n,))
        y = my_f(x, w)
        return x, y

    w = 8.5
    for i in range(500):
        print("Iteration {}".format(i))
        x, y = generate_batch(w)
        model.fit(x, y)
        print(df.get_weights()[0])

    x = np.arange(-1, w + 1, 0.01)
    y_true = my_f(x, w)
    y_pred = model.predict(x)

    import matplotlib.pyplot as plt
    plt.plot(x, y_true)
    plt.plot(x, y_pred)
    plt.show()


# Create now our dynamic depth neural network. It should learn to do the XOR operation
from keras.layers import Dense, Lambda, multiply, RepeatVector, add, Activation, Reshape, BatchNormalization, Flatten
from keras.regularizers import l1, l2

N_max = 10
units = 5
layer_count_reg = l1(0.005)
dense_reg=l2(0.001)

nw_input = Input((2,))
nw = nw_input
nw = Dense(units, trainable=False)(nw)

dummy_layer = Dense(1, trainable=False, kernel_initializer='zeros')(nw)
def constant(c):
    return Lambda(lambda x: x * 0 + c)(dummy_layer)

df = DeltaF(alpha_regularizer=layer_count_reg)

for i in range(N_max):
    dft = df(constant(i))
    dft = RepeatVector(units)(dft)
    dft = Reshape((units,))(dft)

    di = Dense(units, kernel_regularizer=dense_reg)(nw)
    # di = BatchNormalization()(di)
    tmp0 = multiply([di, dft])

    if True:
        tmp1 = nw
    else:
        dft_i = Lambda(lambda x: 1 - x)(dft)
        tmp1 = multiply([dft_i, nw])

    tmp = add([tmp0, tmp1])
    nw = Activation('relu')(tmp)

    # nw = Lambda(lambda fc: K.relu(nw + fc * df(constant(i))))(fc)
nw = Dense(1, name="d_final", activation='sigmoid', trainable=False)(nw)

model = Model(nw_input, nw)
model.compile('adadelta', mean_squared_error, metrics=['binary_accuracy'])

# model.summary()

def generate_data(n):
    x = np.random.uniform(0, 1, (n, 2))
    x[x>=.5] = 1.
    x[x<.5] = 0.
    y = x[:, 0] + x[:, 1]
    y[y > 1] = 0.
    return x, y

if False: # just for testing
    for i in range(10000):
        print("Iteration {}".format(i))
        x, y = generate_data(1000)
        model.fit(x, y)
        print("w={}".format(df.get_weights()[0][0]))


# Das proof-of-concept ist gemacht. Jetzt muss noch eine sauberere Implementierung daher:)


class ReluRegularizer(regularizers.Regularizer):
    """Regularizer that works like relu"""
    def __init__(self, l=0., pow=1.0):
        self.l = K.cast_to_floatx(l)
        self.pow = K.cast_to_floatx(pow)

    def __call__(self, x):
        regularization = 0.
        if self.l:
            regularization += K.sum(K.relu(K.pow(x, self.pow)))
        return regularization

    def get_config(self):
        return {'l': float(self.l), 'pow':self.pow}


def dynamic_layer_count(input_layer, layer_builder, n_penalty=ReluRegularizer(0.005), res_net=True, batch_norm=False, max_layer_count=10, initial_layer_count=1.):
    # max_layer_count can be removed in a real implementation

    # Get the input dimensions
    nw = input_layer
    dim = list(map(lambda x: int(str(x)), nw._keras_shape[1:]))
    dim_n = np.prod(dim)

    # Create a function that may be used to create constants (as keras tensors)
    dummy_layer = input_layer
    if len(dim) > 1:
        dummy_layer = Flatten()(dummy_layer)
    dummy_layer = Dense(1, kernel_initializer='zeros', trainable=False)(dummy_layer)
    def constant(c):
        return Lambda(lambda x: 0 * x + c)(dummy_layer)

    # Create a layer taht controls the count of layers to add
    deltaF = DeltaF(alpha_regularizer=n_penalty, default_layer_count=initial_layer_count)

    # Create now the dynamic layer count. Here we use a fixed count of layers, but most of them
    # will be zero
    for i in range(max_layer_count):

        # Get the factor for the current layer (the ith layer)
        con = constant(i)
        lf = deltaF(con)

        # Resize it to the required dimension
        lf = RepeatVector(dim_n)(lf)
        lf = Reshape(dim)(lf)

        # Calculate the next layer
        l_next = layer_builder(nw)

        # If required: Do a batch normalization
        if batch_norm:
            l_next = BatchNormalization()(l_next)

        # Weight the new calculation
        l_next = multiply([l_next, lf])

        # If required: Weight the old calculation
        if not res_net:
            lf_i = Lambda(lambda x: 1 - x)(lf)
            nw = multiply([nw, lf_i])

        # Sum up the two values:
        nw = add([l_next, nw])

        # Use a non-linearity. Important: The equation f(f(x))=f(x) MUST be true for this non-linearity
        nw = Activation('relu')(nw)

    # Everything is done:)
    return nw, deltaF

def dynamic_layer_count_ext(input_layer, layer_builder, n_penality=ReluRegularizer(0.005), res_net=True, batch_norm=False, max_layer_count=10, init_activation=True):
    nw = layer_builder(input_layer)
    if init_activation:
        nw = Activation('relu')(nw)
    if batch_norm:
        nw = BatchNormalization()(nw)
    return dynamic_layer_count(nw, layer_builder, n_penality, res_net, batch_norm, max_layer_count, 0.5)

# Dummy model: Recreate the XOR-model
units = 5

nw_input = Input((2,))
nw = nw_input

# Reshape the current representation to "units" neurons
# nw = Dense(units, trainable=False)(nw)

# Use a dynamic layer count (fully connected layers)
nw, deltaF = dynamic_layer_count_ext(nw, lambda l: Dense(units, kernel_regularizer=l2(0.001))(l), max_layer_count=4)

# Create an output layer
nw = Dense(1, activation='sigmoid', trainable=False)(nw)

model = Model(nw_input, nw)
model.compile(
    optimizer='adadelta',
    loss='binary_crossentropy', # test binary_crossentropy
    metrics=['binary_accuracy']
)

print("w={}".format(deltaF.get_weights()[0][0]))
if False:
    for i in range(1000):
        print("Iteration {}".format(i))
        x, y = generate_data(1000)

        # Define all y=0, this should require less layers
        y *= 0.

        model.fit(x, y, batch_size=1000)
        print("w={}".format(deltaF.get_weights()[0][0]))

# Do an extended test: Test if a CNN for

from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from impl.data.misc import flowers102, birds200
from keras.losses import hinge

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = flowers102.load_data((48, 48))
(x_train, y_train), (x_test, y_test) = birds200.load_data((48, 48))

num_classes = np.prod(np.unique(y_train).shape)
print("num_classes={}".format(num_classes))

data_shape = x_train[0].shape
if len(data_shape) < 3:
    data_shape += (1,)

# Create a simple CNN

# The regularizer values are very (!) important
reg=l2(1e-2)
n_penalty=l2(1e-11)
batch_norm=True
max_layer_count=10

nw_input = Input(data_shape)
nw = nw_input

nw = Convolution2D(32, (3, 3), trainable=False, padding='same')(nw)
nw, df0 = dynamic_layer_count(nw, lambda x: Dropout(0.25)(Convolution2D(32, (3, 3), kernel_regularizer=reg, padding='same')(x)), max_layer_count=max_layer_count, n_penalty=n_penalty, batch_norm=batch_norm)
nw = MaxPooling2D(pool_size=(2, 2))(nw)
# nw = Dropout(0.5)(nw)

nw = Convolution2D(64, (3, 3), trainable=False, padding='same')(nw)
nw, df1 = dynamic_layer_count(nw, lambda x: Dropout(0.25)(Convolution2D(64, (3, 3), kernel_regularizer=reg, padding='same')(x)), max_layer_count=max_layer_count, n_penalty=n_penalty, batch_norm=batch_norm)
nw = MaxPooling2D()(nw)
# nw = Dropout(0.5)(nw)

nw = Flatten()(nw)

nw = Dense(256, trainable=False)(nw)
nw, df2 = dynamic_layer_count(nw, lambda x: Dropout(0.25)(Dense(256, kernel_regularizer=reg)(x)), max_layer_count=max_layer_count, n_penalty=n_penalty, batch_norm=batch_norm)

nw = Dense(num_classes, trainable=False, name="classification", activation='softmax')(nw)

model = Model(nw_input, nw)
model.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

def print_layer_counts():
    print("df0: {}".format(df0.get_weights()[0][0]))
    print("df1: {}".format(df1.get_weights()[0][0]))
    print("df2: {}".format(df2.get_weights()[0][0]))
    print()

import keras.utils

# Test MNIST (see: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) :)
x_train = x_train.reshape(x_train.shape[0], *data_shape)
x_test = x_test.reshape(x_test.shape[0], *data_shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print_layer_counts()
for i in range(10000):
    print("Iteration {}".format(i))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=500, epochs=1)
    print_layer_counts()

# Resultate:
# MNIST (batch size=500):
# Iteration 25
# Train on 60000 samples, validate on 10000 samples
# Epoch 1/1
# 60000/60000 [==============================] - ETA: 0s - loss: 0.0642 - categorical_accuracy: 0.9914 - val_loss: 0.0690 - val_categorical_accuracy: 0.9927
# df0: 0.7753986716270447
# df1: 1.335105061531067
# df2: 1.8371177911758423
# CIFAR10 (batch size=500):
# Iteration 254
# Train on 50000 samples, validate on 10000 samples
# Epoch 1/1
# 50000/50000 [==============================] - 25s 491us/step - loss: 0.6033 - categorical_accuracy: 0.8595 - val_loss: 1.1314 - val_categorical_accuracy: 0.6955
# df0: 0.5839430093765259
# df1: 1.8853037357330322
# df2: 2.000136375427246
# CIFAR100 (nicht ganz fertig) (batch size=500):
# Iteration 166
# Train on 50000 samples, validate on 10000 samples
# Epoch 1/1
# 50000/50000 [==============================] - 24s 490us/step - loss: 2.0777 - categorical_accuracy: 0.5426 - val_loss: 2.9357 - val_categorical_accuracy: 0.3750
# df0: 0.6286575794219971
# df1: 1.7639727592468262
# df2: 3.9990553855895996
# Birds200 (batch size=500):
# 5994/5994 [==============================] - 13s 2ms/step - loss: 0.5885 - categorical_accuracy: 0.9511 - val_loss: 7.3716 - val_categorical_accuracy: 0.1068
# df0: 0.8518016338348389
# df1: 1.1794722080230713
# df2: 5.998287677764893



# TODO:
# Ist regularisierung für den dense / conv layer notwendig?
# f_{i+1}(x) könnte man noch wie folgt regularisieren (so sind die Werte immer normalisiert (aktuell könnten sie theoretisch explodieren):
# g_{i+1}(x) = Relu(f_{i}(x)+d(i)*Layer(f_{i}(x))) # dies entspricht dem "alten" f_{i+1}
# f_{i+1}(x) = d(i)*BatchNorm(g_{i+1}(x)) + (1-d(i))*g_{i+1}(x)