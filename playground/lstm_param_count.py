from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional, Dense


nw_input = Input((10, 10))
nw = nw_input
nw = LSTM(10)(nw)
Model(nw_input, nw).summary()



nw_input = Input((10, 10))
nw = nw_input
nw = Bidirectional(LSTM(10))(nw)
Model(nw_input, nw).summary()