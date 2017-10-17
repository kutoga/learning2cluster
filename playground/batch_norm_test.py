from keras.models import Model
from keras.layers import Dense, TimeDistributed, BatchNormalization, Input, Reshape
from keras.losses import mean_squared_error

nw_input = Input((2,))
nw = nw_input
nw = Reshape((2, 1))(nw)
nw = TimeDistributed(Dense(1))(nw)
nw = TimeDistributed(BatchNormalization())(nw)

model = Model(nw_input, nw)
model.compile('SGD', mean_squared_error)



