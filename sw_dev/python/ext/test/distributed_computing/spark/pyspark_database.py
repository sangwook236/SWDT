#!/usr/bin/env python

from pyspark.sql import SparkSession
import traceback, sys

# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#jdbc-to-other-databases

def mysql_jdbc():
	spark = SparkSession.builder.appName('mysql-jdbc').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	jdbc_df = spark.read \
		.format('jdbc') \
		.option('url', 'jdbc:mysql://host:3306/dbname?characterEncoding=UTF-8&serverTimezone=UTC') \
		.option('driver', 'com.mysql.cj.jdbc.Driver') \
		.option('dbtable', 'tablename') \
		.option('user', 'username') \
		.option('password', 'password') \
		.load()

	jdbc_df.show()

def main():
	mysql_jdbc()

#%%------------------------------------------------------------------

# Usage:
#	python pyspark_machine_learning.py
#	spark-submit --packages mysql:mysql-connector-java:8.0.12 pyspark_machine_learning.py
#	spark-submit --master local[4] --packages mysql:mysql-connector-java:8.0.12 pyspark_machine_learning.py
#	spark-submit --master spark://host:7077 --packages mysql:mysql-connector-java:8.0.12 --executor-memory 10g pyspark_machine_learning.py

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
