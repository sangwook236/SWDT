#---------------------------------------------------------------------
# VGG-like convnet.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()

# Input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# This applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0., nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Generate dummy data.
import numpy as np

x_train = np.random.random((1000, 100, 100, 3))
y_train = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical on-hot encoding.
y_train = keras.utils.to_categorical(y_train, num_classes=10)

model.fit(x_train, y_train, batch_size=32, epochs=10)

#---------------------------------------------------------------------
# Sequence classification with 1D convolutions.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

seq_length = 1000

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

# Generate dummy data.
import numpy as np

x_train = np.random.random((500, seq_length, 100))
y_train = np.random.randint(2, size=(500, 1))

model.fit(x_train, y_train, batch_size=16, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=16)
