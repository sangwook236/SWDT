# REF [site] >> https://pypi.python.org/pypi/automl/2.7.7

#-----------------------------------------------------------

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

df_train, df_test = get_boston_dataset()

column_descriptions = {
	'MEDV': 'output',
	'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

ml_predictor.score(df_test, df_test.MEDV)

#-----------------------------------------------------------

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

# Load data
df_train, df_test = get_boston_dataset()

# Tell auto_ml which column is 'output'.
# Also note columns that aren't purely numerical.
# Examples include ['nlp', 'date', 'categorical', 'ignore'].
column_descriptions = {
	'MEDV': 'output',
	'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

# Score the model on test data.
test_score = ml_predictor.score(df_test, df_test.MEDV)

# auto_ml is specifically tuned for running in production.
# It can get predictions on an individual row (passed in as a dictionary).
# A single prediction like this takes ~1 millisecond.
# Here we will demonstrate saving the trained model, and loading it again.
file_name = ml_predictor.save()

trained_model = load_ml_model(file_name)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predictions = trained_model.predict(df_test)
print(predictions)
