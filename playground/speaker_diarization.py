from keras.models import Model
from keras.layers import Input, Conv2D, Conv1D, Flatten, TimeDistributed, MaxPool2D, Dropout, Dense, RepeatVector

# Can be changed
input_length = 4 * 128
class_count = 10

assert input_length % 4 == 0
n_f = 128
nw_input = Input((input_length, n_f, 1))
nw = nw_input
nw = Conv2D(32, (3, 3), padding='same')(nw)
nw = MaxPool2D()(nw)
nw = Conv2D(64, (3, 3), padding='same')(nw)
nw = MaxPool2D()(nw)
nw = TimeDistributed(Flatten())(nw)
nw = Conv1D(512, 15, padding='same')(nw)
nw = Conv1D(256, 1, padding='same')(nw)
nw = TimeDistributed(Dense(class_count, activation='softmax'))(nw)
# nw = RepeatVector(4)(nw)

model = Model(nw_input, nw)

model.summary()
