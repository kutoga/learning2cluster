import keras.backend as K
import numpy as np

# See: https://github.com/fchollet/keras/issues/2226

X = K.placeholder(ndim=1) #specify the right placeholder
Y = K.relu(K.sign(X)) # loss function
fn = K.function([X], K.gradients(Y, [X])) #function to call the gradient
fn2 = K.function([X], [Y]) #function to call the gradient

def calc(x):
    x_in = [np.asarray([x])]
    print("f({})  = {}".format(x, fn2(x_in)))
    print("f'({}) = {}".format(x, fn(x_in)))
    print()

calc(-0.01)
calc(0.0)
calc(0.01)
