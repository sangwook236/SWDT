#%%-------------------------------------------------------------------
# Sequence classification with LSTM.

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np

max_features = 1000

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data.
x_train = np.random.random((500, 100))
y_train = np.random.randint(2, size=(500, 1))

model.fit(x_train, y_train, batch_size=16, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=16)

#%%-------------------------------------------------------------------
# Stacked LSTM for sequence classification.

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# Expected input data shape: (batch_size, timesteps, data_dim).
model = Sequential()
# To stack recurrent layers, use return_sequences=True on any recurrent layer that feeds into another recurrent layer.
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # Return a sequence of vectors of dimension 32.
model.add(LSTM(32, return_sequences=True))  # Return a sequence of vectors of dimension 32.
model.add(LSTM(32))  # Return a single vector of dimension 32.
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy training data.
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data.
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
		batch_size=64, epochs=5,
		validation_data=(x_val, y_val))

#%%-------------------------------------------------------------------
# Stateful stacked LSTM model.

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim).
# Note that we have to provide the full batch_input_shape since the network is stateful.
# The sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy training data.
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data.
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
		batch_size=batch_size, epochs=5, shuffle=False,
		validation_data=(x_val, y_val))
