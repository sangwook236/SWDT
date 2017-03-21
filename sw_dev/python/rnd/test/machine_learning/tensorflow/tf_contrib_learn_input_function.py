# REF [site] >> https://www.tensorflow.org/get_started/input_fn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
housing_dataset_dir = "D:/dataset/pattern_recognition/uci/housing_data_set"
housing_training_dataset = housing_dataset_dir + "/boston_train.csv"
housing_test_dataset = housing_dataset_dir + "/boston_test.csv"
housing_prediction_dataset = housing_dataset_dir + "/boston_predict.csv"
housing_model_dir = housing_dataset_dir + "/housing_model"

# Importg the housing data.
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv(housing_training_dataset, skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv(housing_test_dataset, skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv(housing_prediction_dataset, skipinitialspace=True, skiprows=1, names=COLUMNS)

# Define feature columns.
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Create the regressor.
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10], model_dir=housing_model_dir)

# Train the regressor.
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

# Evaluate the model.
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)

loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

# Make predictions.
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions.

predictions = list(itertools.islice(y, 6))
print ("Predictions: {}".format(str(predictions)))

# Build the input_fn.
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
	labels = tf.constant(data_set[LABEL].values)
	return feature_cols, labels
