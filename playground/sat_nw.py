# The described network can be used to "solve" the SAT problem
# The encoding is a bit ugly, but jeah;)...

from keras.models import Model
from keras.layers import Input, Dense, Lambda, multiply

import keras.backend as K

nw_input = Input((1,))
nw = nw_input

def f_or(*x_l):
    return multiply(x_l)

def f_and(*x_l):
    return Lambda(lambda tmp: K.max(x_l))(nw_input)

def f_not(x):
    return Lambda(lambda x: 1 - x)(x)

def var(activation='sigmoid'):
    tmp = Lambda(lambda x: x * 0 + 1)(nw_input)
    tmp = Dense(1, use_bias=False)(tmp)
    tmp = Dense(activation=activation)
    return tmp

##### The SAT problem definition ######
x1 = var()
x2 = var()
x3 = var()
x4 = var()

# Example from https://de.wikipedia.org/wiki/3-SAT
F = f_and(
    f_or(f_not(x1), x2, x3),
    f_or(x2, f_not(x3), x4),
    f_or(x1, f_not(x2))
)
