# REF [site] >> http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

#%%-------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#%%-------------------------------------------------------------------

def display_history(history):
	# List all data in history.
	print(history.history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)
	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

#%%-------------------------------------------------------------------
# Prepare dataset.

# Fix random seed for reproducibility.
np.random.seed(7)

# Load the dataset but only keep the top n words, zero the rest.
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Truncate and pad input sequences so that they are all the same length for modeling.
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vecor_length = 32

#%%-------------------------------------------------------------------
# Simple LSTM for sequence classification.

# Create the model.
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
display_history(history)

# Final evaluation of the model.
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

#%%-------------------------------------------------------------------
# LSTM for sequence classification with dropout.

# Dropout can be applied between layers.
#	- Dropout is a powerful technique for combating overfitting in your LSTM models.
#	- The dropout has the desired impact on training with a slightly slower trend in convergence and in this case a lower final accuracy.

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
display_history(history)

# Final evaluation of the model.
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

#%%-------------------------------------------------------------------
# LSTM for sequence classification with dropout.

# Dropout can be applied to the input and recurrent connections of the memory units with the LSTM precisely and separately.
#	- Keras provides this capability with parameters on the LSTM layer, the 'dropout' for configuring the input dropout and 'recurrent_dropout' for configuring the recurrent dropout.
#	- The LSTM specific dropout has a more pronounced effect on the convergence of the network than the layer-wise dropout.
#	- You may bet better results with the gate-specific dropout.

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
display_history(history)

# Final evaluation of the model.
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

#%%-------------------------------------------------------------------
# LSTM and convolutional neural network (CNN) for sequence classification.

# Convolutional neural networks excel at learning the spatial structure in input data.
# The IMDB review data does have a one-dimensional spatial structure in the sequence of words in reviews.

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
display_history(history)

# Final evaluation of the model.
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))
