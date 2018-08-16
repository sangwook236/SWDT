#!/usr/bin/env spark-submit

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.types as types
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
import traceback, sys

def describe_statistics():
	spark = SparkSession.builder.appName('describe-statistics').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	sc = spark.sparkContext
	sc.setLogLevel('WARN')

	# Read a dataset in and remove its header.
	fraud = sc.textFile('dataset/ccFraud.csv.gz')
	header = fraud.first()

	fraud = fraud.filter(lambda row: row != header).map(lambda row: [int(elem) for elem in row.split(',')])

	# Create a schema.
	fields = [*[types.StructField(h[1:-1], types.IntegerType(), True) for h in header.split(',')]]
	schema = types.StructType(fields)

	# Create a DataFrame.
	fraud_df = spark.createDataFrame(fraud, schema)
	fraud_df.printSchema()

	# For a better understanding of categorical columns.
	fraud_df.groupby('gender').count().show()

	# Describe for the truly numerical features.
	numerical = ['balance', 'numTrans', 'numIntlTrans']
	desc = fraud_df.describe(numerical)
	desc.show()

	# Check the skeweness.
	fraud_df.agg({'balance': 'skewness'}).show()

	# Check the correlation between the features.
	fraud_df.corr('balance', 'numTrans')

	# Create a correlations matrix.
	n_numerical = len(numerical)
	corr = []
	for i in range(0, n_numerical):
		temp = [None] * i
		for j in range(i, n_numerical):
			temp.append(fraud_df.corr(numerical[i], numerical[j]))
		corr.append(temp)

def visualize_data():
	#%matplotlib inline
	plt.style.use('ggplot')
	#output_notebook()

	spark = SparkSession.builder.appName('visualize-data').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	sc = spark.sparkContext
	sc.setLogLevel('WARN')

	# Read a dataset in and remove its header.
	fraud = sc.textFile('dataset/ccFraud.csv.gz')
	header = fraud.first()

	fraud = fraud.filter(lambda row: row != header).map(lambda row: [int(elem) for elem in row.split(',')])

	# Create a schema.
	fields = [*[types.StructField(h[1:-1], types.IntegerType(), True) for h in header.split(',')]]
	schema = types.StructType(fields)

	# Create a DataFrame.
	fraud_df = spark.createDataFrame(fraud, schema)

	# Histogram.
	hists = fraud_df.select('balance').rdd.flatMap(lambda row: row).histogram(20)

	# Plot a histogram (#1).
	data = {
		'bins': hists[0][:-1],
		'freq': hists[1]
	}

	plt.bar(data['bins'], data['freq'], width=2000)
	plt.title('Histogram of "balance"')
	plt.show()

	#p = figure(plot_height=600, plot_width=600, title='Histogram of "balance"', x_axis_label='bins', y_axis_label='freq')
	#p.quad(bottom=0, top=data['bins'], left=data['freq'], right=None, fill_color='red', line_color='black')
	#show(p)

	# Plot a histogram (#2).
	data_driver = {'obs': fraud_df.select('balance').rdd.flatMap(lambda row: row).collect()}
	plt.hist(data_driver['obs'], bins=20)
	plt.title('Histogram of "balance" using .hist()')
	plt.show()

	# Sample our dataset at 0.02%.
	numerical = ['balance', 'numTrans', 'numIntlTrans']
	data_sample = fraud_df.sampleBy('gender', {1: 0.0002, 2: 0.0002}).select(numerical)

	# Plot a scatter plot.
	data_multi = dict([(elem, data_sample.select(elem).rdd.flatMap(lambda row: row).collect()) for elem in numerical])
	output_file('scatter.html')
	p = figure(plot_width=400, plot_height=400)
	p.circle(data_multi['balance'], data_multi['numTrans'], size=3, color='navy', alpha=0.5)
	p.xaxis.axis_label = 'balance'
	p.yaxis.axis_label = 'numTrans'
	show(p)

def main():
	describe_statistics()
	visualize_data()

#%%------------------------------------------------------------------

# Usage:
#	spark-submit pyspark_descriptive_statistics.py
#	spark-submit --master local[4] pyspark_descriptive_statistics.py
#	spark-submit --master spark://host:7077 --executor-memory 10g pyspark_descriptive_statistics.py

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
