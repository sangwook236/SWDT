#!/usr/bin/env spark-submit

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as func
import traceback, sys

def handle_duplicate():
	spark = SparkSession.builder.appName('handle-duplicate').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()

	df = spark.createDataFrame(
		[
			(1, 144.5, 5.9, 33, 'M'),
			(2, 167.2, 5.4, 45, 'M'),
			(3, 124.1, 5.2, 23, 'F'),
			(4, 144.5, 5.9, 33, 'M'),
			(5, 133.2, 5.7, 54, 'F'),
			(3, 124.1, 5.2, 23, 'F'),
			(5, 129.2, 5.3, 42, 'M'),
		],
		['id', 'weight', 'height', 'age', 'gender']
	)

	#--------------------
	print('Count of rows: {0}'.format(df.count()))
	print('Count of distinct rows: {0}'.format(df.distinct().count()))

	df = df.dropDuplicates()
	df.show()

	#--------------------
	print('Count of ids: {0}'.format(df.count()))
	print('Count of distinct ids: {0}'.format(
		df.select([c for c in df.columns if c != 'id']).distinct().count())
	)

	df = df.dropDuplicates(subset=[c for c in df.columns if c != 'id'])

	#--------------------
	df.agg(
		func.count('id').alias('count'),
		func.countDistinct('id').alias('distinct')
	).show()

def handle_missing_value():
	spark = SparkSession.builder.appName('handle-missing-value').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()

	df_miss = spark.createDataFrame(
		[
			(1, 143.5, 5.6, 28, 'M', 100000),
			(2, 167.2, 5.4, 45, 'M', None),
			(3, None , 5.2, None, None, None),
			(4, 144.5, 5.9, 33, 'M', None),
			(5, 133.2, 5.7, 54, 'F', None),
			(6, 124.1, 5.2, None, 'F', None),
			(7, 129.2, 5.3, 42, 'M', 76000),
		],
		['id', 'weight', 'height', 'age', 'gender', 'income']
	)

	# Find the number of missing observations per row.
	df_miss.rdd.map(lambda row: (row['id'], sum([c == None for c in row]))).collect()

	# Check what percentage of missing observations are there in each column.
	df_miss.agg(*[
		(1 - (func.count(c) / func.count('*'))).alias(c + '_missing')
		for c in df_miss.columns
	]).show()

	# Drop the 'income' feature.
	df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])

	# Drop the observations.
	df_miss_no_income.dropna(thresh=3).show()

	# Impute the observations.
	means = df_miss_no_income.agg(
		*[func.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']
	).toPandas().to_dict('records')[0]
	means['gender'] = 'missing'
	df_miss_no_income.fillna(means).show()

def handle_outlier():
	spark = SparkSession.builder.appName('handle-missing-value').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()

	df_outliers = spark.createDataFrame(
		[
			(1, 143.5, 5.3, 28),
			(2, 154.2, 5.5, 45),
			(3, 342.3, 5.1, 99),
			(4, 144.5, 5.5, 33),
			(5, 133.2, 5.4, 54),
			(6, 124.1, 5.1, 21),
			(7, 129.2, 5.3, 42),
		],
		['id', 'weight', 'height', 'age']
	)

	# Calculate the lower and upper cut off points for each feature.
	cols = ['weight', 'height', 'age']
	bounds = {}

	for col in cols:
		quantiles = df_outliers.approxQuantile(col, [0.25, 0.75], 0.05)
		IQR = quantiles[1] - quantiles[0]
		bounds[col] = [quantiles[0] - 1.5 * IQR, quantiles[1] + 1.5 * IQR]

	outliers = df_outliers.select(*['id'] + [
		(
			(df_outliers[c] < bounds[c][0]) |
			(df_outliers[c] > bounds[c][1])
		).alias(c + '_o') for c in cols
	])
	outliers.show()

	# List the values significantly differing from the rest of the distribution.
	df_outliers = df_outliers.join(outliers, on='id')
	df_outliers.filter('weight_o').select('id', 'weight').show()
	df_outliers.filter('age_o').select('id', 'age').show()

def main():
	handle_duplicate()
	handle_missing_value()
	handle_outlier()

#%%------------------------------------------------------------------

# Usage:
#	spark-submit pyspark_preprocessing.py
#	spark-submit --master local[4] pyspark_preprocessing.py
#	spark-submit --master spark://host:7077 --executor-memory 10g pyspark_preprocessing.py

if '__main__' == __name__:
	try:
		main()
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
