#!/usr/bin/env spark-submit

from pyspark.sql import SparkSession
import tensorflow as tf
import tensorframes as tfs
from pyspark.sql import Row
import traceback, sys

def simple_example_1():
	spark = SparkSession.builder.appName('simple-tensorframes-example-1').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	rdd = [Row(x=float(x)) for x in range(10)]
	df = spark.createDataFrame(rdd)

	df.show()

	# Execute the tensor graph.
	with tf.Graph().as_default() as graph:
		# A block placeholder.
		x = tfs.block(df, 'x')
		z = tf.add(x, 3, name='z')

		# Tensor -> dataframe.
		df2 = tfs.map_blocks(z, df)

	print('z =', z)
	df2.show()

def simple_example_2():
	spark = SparkSession.builder.appName('simple-tensorframes-example-2').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	rdd = [Row(y=[float(y), float(-y)]) for y in range(10)]
	df = spark.createDataFrame(rdd)

	df.show()
	tfs.print_schema(df)

	# Analyze first to find the dimensions of the vectors.
	df2 = tfs.analyze(df)

	tfs.print_schema(df2)

	# Make a copy of the 'y' column: An inexpensive operation in Spark 2.0+.
	df3 = df2.select(df2.y, df2.y.alias('z'))

	# Execute the tensor graph.
	with tf.Graph().as_default() as graph:
		y_input = tfs.block(df3, 'y', tf_name='y_input')
		z_input = tfs.block(df3, 'z', tf_name='z_input')

		# Perform elementwise sum and minimum.
		y = tf.reduce_sum(y_input, [0], name='y')
		z = tf.reduce_min(z_input, [0], name='z')

		(data_sum, data_min) = tfs.reduce_blocks([y, z], df3)

	print('Elementwise sum: %s and minimum: %s' % (data_sum, data_min))

def main():
	simple_example_1()
	simple_example_2()

#%%------------------------------------------------------------------

# Usage:
#	spark-submit --packages databricks:tensorframes:0.4.0-s_2.11 pyspark_tensorframes.py
#	spark-submit --master local[4] --packages databricks:tensorframes:0.4.0-s_2.11 pyspark_tensorframes.py
#	spark-submit --master spark://host:7077 --packages databricks:tensorframes:0.4.0-s_2.11 --executor-memory 10g pyspark_tensorframes.py

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
