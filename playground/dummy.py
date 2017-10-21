
# Just trying to help a user:
# https://github.com/fchollet/keras/issues/8207

from keras.models import Sequential
from keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Flatten

n_classes=10
epoch_len = 3
n_epochs = 10

model = Sequential()
model.add(TimeDistributed(Bidirectional(LSTM(11, return_sequences=True, recurrent_dropout=0.1, unit_forget_bias=True), input_shape=(3, 3, epoch_len), merge_mode='sum'), input_shape=(n_epochs, 3, epoch_len)))
model.add(TimeDistributed(Dense(7)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(12, return_sequences=True, recurrent_dropout=0.1, unit_forget_bias=True), merge_mode='sum'))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
