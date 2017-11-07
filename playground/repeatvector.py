from keras.models import Model
from keras.layers import Input, RepeatVector, Lambda
import keras.backend as K

nw_input = Input((28, 28, 1))
nw = nw_input
nw = Lambda(lambda x: K.repeat_elements(nw, 2, axis=3))(nw)
# nw = RepeatVector(2)(nw)

Model(nw_input, nw).summary()
