import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#%%-------------------------------------------------------------------
# For a single-input model with 2 classes (binary classification).

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['accuracy'])

# Generate dummy data.
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))

# Train the model, iterating on the data in batches of 32 samples.
model.fit(data, labels, epochs=10, batch_size=32)

#%%-------------------------------------------------------------------
# For a single-input model with 10 classes (categorical classification).

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])

# Generate dummy data.
import numpy as np

data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical on-hot encoding.
binary_labels = keras.utils.to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples.
model.fit(data, binary_labels, epochs=10, batch_size=32)

#%%-------------------------------------------------------------------
# Multilayer Perceptron (MLP) for multi-class softmax classification.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# In the first layer, you must specify the expected input data shape: here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
	optimizer=sgd,
	metrics=['accuracy'])

# Generate dummy data.
import numpy as np

x_train = np.random.random((1000, 20))
y_train_tmp = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical on-hot encoding.
y_train = keras.utils.to_categorical(y_train_tmp, num_classes=10)

model.fit(x_train, y_train,
	epochs=20,
	batch_size=128)

#score = model.evaluate(x_test, y_test, batch_size=128)
