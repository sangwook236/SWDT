#!/usr/bin/env python

from pyspark.sql import SparkSession
import pyspark.sql.types as types
import traceback, sys

# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#jdbc-to-other-databases

def sqlite_jdbc():
	spark = SparkSession.builder.appName('sqlite-jdbc') \
		.config('spark.jars.packages', 'org.xerial:sqlite-jdbc:3.23.1') \
		.getOrCreate()
	#spark = SparkSession.builder.appName('sqlite-jdbc') \
	#	.config('spark.jars', 'sqlite-jdbc-3.23.1.jar') \
	#	.getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	if False:
		#db_url = 'jdbc:sqlite:/path/to/dbfile'  # File DB.
		df = spark.read \
			.format('jdbc') \
			.option('url', 'jdbc:sqlite:iris.db') \
			.option('driver', 'org.sqlite.JDBC') \
			.option('dbtable', 'iris') \
			.load()
	elif False:
		# REF [site] >> https://www.sqlite.org/inmemorydb.html
		#db_url = 'jdbc:sqlite::memory:'  # In-memory DB.
		db_url = 'jdbc:sqlite::memory:?cache=shared'  # Shared in-memory DB.
		#db_url = 'jdbc:sqlite:dbname?mode=memory&cache=shared'  # Named, shared in-memory DB.

		# NOTE [error] >> Requirement failed: Option 'dbtable' is required.
		# NOTE [error] >> SQL error or missing database (no such table: test123).
		df = spark.read \
			.format('jdbc') \
			.option('url', db_url) \
			.option('driver', 'org.sqlite.JDBC') \
			.option('dbtable', 'test123') \
			.load()
	else:
		rdd = spark.sparkContext.parallelize([
			(123, 'Katie', 19, 'brown'),
			(234, 'Michael', 22, 'green'),
			(345, 'Simone', 23, 'blue')
		])
		# Specify schema.
		schema = types.StructType([
			types.StructField('id', types.LongType(), True),
			types.StructField('name', types.StringType(), True),
			types.StructField('age', types.LongType(), True),
			types.StructField('eyeColor', types.StringType(), True)
		])
		df = spark.createDataFrame(rdd, schema)
	df.show()

	#df.createOrReplaceTempView('swimmers')
	#df.printSchema()

	# NOTE [info] >> It seems that Only file DB of SQLite can be used in Spark.
	db_url = 'jdbc:sqlite:test.sqlite'  # File DB.
	#df.write.jdbc(url=db_url, table='test', mode='overwrite', properties={'driver': 'org.sqlite.JDBC'})
	# Isolation level: NONE, READ_COMMITTED, READ_UNCOMMITTED, REPEATABLE_READ, SERIALIZABLE.
	#	REF [site] >> https://stackoverflow.com/questions/16162357/transaction-isolation-levels-relation-with-locks-on-table
	df.write \
		.format('jdbc') \
		.mode('overwrite') \
		.option('url', db_url) \
		.option('driver', 'org.sqlite.JDBC') \
		.option('dbtable', 'swimmers') \
		.option('isolationLevel', 'NONE') \
		.save()

def mysql_jdbc():
	spark = SparkSession.builder.appName('mysql-jdbc') \
		.config('spark.jars.packages', 'mysql:mysql-connector-java:8.0.12') \
		.getOrCreate()
	#spark = SparkSession.builder.appName('mysql-jdbc') \
	#	.config('spark.jars', 'mysql-connector-java-8.0.12-bin.jar') \
	#	.getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	df = spark.read \
		.format('jdbc') \
		.option('url', 'jdbc:mysql://host:3306/dbname?characterEncoding=UTF-8&serverTimezone=UTC') \
		.option('driver', 'com.mysql.cj.jdbc.Driver') \
		.option('dbtable', 'tablename') \
		.option('user', 'username') \
		.option('password', 'password') \
		.load()
	df.show()

def main():
	sqlite_jdbc()
	#mysql_jdbc()

#%%------------------------------------------------------------------

# Usage:
#	python pyspark_database.py
#	spark-submit --packages mysql:mysql-connector-java:8.0.12,org.xerial:sqlite-jdbc:3.23.1 pyspark_database.py
#	spark-submit --master local[4] --packages mysql:mysql-connector-java:8.0.12,org.xerial:sqlite-jdbc:3.23.1 pyspark_database.py
#	spark-submit --master spark://host:7077 --packages mysql:mysql-connector-java:8.0.12,org.xerial:sqlite-jdbc:3.23.1 --executor-memory 10g pyspark_database.py

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
