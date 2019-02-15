#!/usr/bin/env python

# REF [site] >>
#	https://dask.org/
#	https://github.com/dask/dask
#	https://github.com/dask/distributed
#	https://github.com/dask/dask-ml

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
from dask_ml.linear_model import LogisticRegression
from dask_ml.datasets import make_classification
import dask_glm.algorithms
import dask_glm.families
import dask_glm.regularizers
from dask_glm.datasets import make_regression
from dask.distributed import Client, progress
import os, json
import pandas as pd
from sklearn.linear_model import LinearRegression

# REF [site] >> https://examples.dask.org/array.html
def dask_array_example():
	client = Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='2GB')
	print(client)

	x = da.random.random((10000, 10000), chunks=(1000, 1000))

	y = x + x.T
	z = y[::2, 5000:].mean(axis=1)

	# Call .compute() when you want your result as a NumPy array.
	print(z.compute())

	# If you have the available RAM for your dataset then you can persist data in memory.
	# This allows future computations to be much faster.
	y = y.persist()

	print(y[0, 0].compute())
	print(y.sum().compute())

# REF [site] >> https://examples.dask.org/bag.html
def dask_bag_example():
	client = Client(n_workers=4, threads_per_worker=1)
	print(client)

	os.makedirs('data', exist_ok=True)  # Create data/ directory.

	b = dask.datasets.make_people()  # Make records of people.
	b.map(json.dumps).to_textfiles('data/*.json')  # Encode as JSON, write to disk.

	b = db.read_text('data/*.json').map(json.loads)

	print(b.take(2))
	print(b.filter(lambda record: record['age'] > 30).take(2))  # Select only people over 30.
	print(b.map(lambda record: record['occupation']).take(2))  # Select the occupation field.
	print(b.count().compute())  # Count total number of records.

	# Chain computations.
	# It is common to do many of these steps in one pipeline, only calling compute or take at the end.
	result = (b.filter(lambda record: record['age'] > 30)
		.map(lambda record: record['occupation'])
		.frequencies(sort=True)
		.topk(10, key=1))

	# Transform and Store
	(b.filter(lambda record: record['age'] > 30)  # Select records of interest.
		.map(json.dumps)  # Convert Python objects to text.
		.to_textfiles('data/processed.*.json'))  # Write to local disk.

	# Convert to Dask Dataframes,
	def flatten(record):
		return {
			'age': record['age'],
			'occupation': record['occupation'],
			'telephone': record['telephone'],
			'credit-card-number': record['credit-card']['number'],
			'credit-card-expiration': record['credit-card']['expiration-date'],
			'name': ' '.join(record['name']),
			'street-address': record['address']['address'],
			'city': record['address']['city']
		}

	print(b.take(1))
	print(b.map(flatten).take(1))

	df = b.map(flatten).to_dataframe()
	print(df.head())
	print(df[df.age > 30].occupation.value_counts().nlargest(10).compute())

# REF [site] >> https://examples.dask.org/dataframe.html
def dask_dataframe_example():
	client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
	print(client)

	df = dask.datasets.timeseries()
	print(df)
	print(df.dtypes)

	pd.options.display.precision = 2
	pd.options.display.max_rows = 10
	print(df.head(3))

	# Standard Pandas operations.
	df2 = df[df.y > 0]
	df3 = df2.groupby('name').x.std()
	print(df3)

	# Call .compute() when you want your result as a Pandas dataframe.
	# If you started Client() above then you may want to watch the status page during computation.
	computed_df = df3.compute()
	print(type(computed_df))
	print(computed_df)

	# Persist data in memory
	# If you have the available RAM for your dataset then you can persist data in memory.
	# This allows future computations to be much faster.
	df = df.persist()
	print(df[['x', 'y']].resample('1h').mean().head())
	df[['x', 'y']].resample('24h').mean().compute().plot()
	print(df[['x', 'y']].rolling(window='24h').mean().head())
	print(df.loc['2000-01-05'])
	print(df.loc['2000-01-05'].compute())

	# Set index.
	# Data is sorted by the index column.
	df = df.set_index('name')
	print(df)

	df = df.persist()
	print(df.loc['Alice'].compute())

	# Groupby Apply with scikit-learn.
	# Now that our data is sorted by name we can easily do operations like random access on name, or groupby-apply with custom functions.

	def train(partition):
		est = LinearRegression()
		est.fit(partition[['x']].values, partition.y.values)
		return est

	print(df.groupby('name').apply(train, meta=object).compute())

def simple_example():
	X, y = make_classification(n_samples=10000, n_features=2, chunks=50)

	X = dd.from_dask_array(X, columns=["a","b"])
	y = dd.from_array(y)

	lr = LogisticRegression()
	lr.fit(X.values, y.values)

	print('Predictions =', lr.predict(X.values).compute())
	print('Probabilities =', lr.predict_proba(X.values).compute())
	print('Scores =', lr.score(X.values, y.values).compute())

# REF [site] >> https://examples.dask.org/machine-learning/glm.html
def glm_example():
	client = Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='2GB')

	X, y = make_regression(n_samples=200000, n_features=100, n_informative=5, chunksize=10000)
	X, y = dask.persist(X, y)

	b = dask_glm.algorithms.admm(X, y, max_iter=5)
	print(b)

	b = dask_glm.algorithms.proximal_grad(X, y, max_iter=5)
	print(b)

	family = dask_glm.families.Poisson()
	regularizer = dask_glm.regularizers.ElasticNet()
	b = dask_glm.algorithms.proximal_grad(
		X, y,
		max_iter=5,
		family=family,
		regularizer=regularizer,
	)
	print(b)

# Usage:
#	dask-scheduler
#		Scheduler started at 127.0.0.1:8786
#	dask-worker 127.0.0.1:8786
#	dask-worker 127.0.0.1:8786
#	dask-worker 127.0.0.1:8786
# REF [site] >> https://distributed.readthedocs.io/en/latest/quickstart.html
def distributed_quickstart():
	# At least one dask-worker must be running after launching a scheduler.
	#client = Client('127.0.0.1:8786')  # Launch a Client and point it to the IP/port of the scheduler.
	#client = Client()  # Set up local cluster on your laptop.
	client = Client(n_workers=4, threads_per_worker=1)

	def square(x):
		return x ** 2

	def neg(x):
		return -x

	A = client.map(square, range(10))
	B = client.map(neg, A)

	total = client.submit(sum, B)
	print(total.result())  # Result for single future.

	print(client.gather(A))  # Gather for many futures.

	# When things go wrong, or when you want to reset the cluster state, call the restart method.
	#client.restart()

# REF [site] >> https://examples.dask.org/machine-learning/scale-scikit-learn.html
def sklearn_example():
	raise NotImplementedError

# REF [site] >> http://ml.dask.org/tensorflow.html
def tensorflow_example():
	raise NotImplementedError

# REF [site] >> https://examples.dask.org/machine-learning/tpot.html
def automl_example():
	raise NotImplementedError

def main():
	#dask_array_example()
	dask_bag_example()
	#dask_dataframe_example()

	#simple_example()
	#glm_example()

	#distributed_quickstart()

	#sklearn_example()  # Not yet implemented.
	#tensorflow_example()  # Not yet implemented.

	#automl_example()  # Not yet implemented.

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
