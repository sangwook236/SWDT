#!/usr/bin/env python

# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.types as types
import pyspark.sql.functions as func
import traceback, sys

# Spark configuration:
#	REF [site] >> https://spark.apache.org/docs/latest/configuration.html
#	1. Configurations declared explicitly in the user's code. (highest priority)
#	2. Flags passed to spark-submit.
#	3. Values in the properties file (conf/spark-defaults.conf in the Spark directory).
#	4. Default values. (lowest priority)

# Master URLs:
#	REF [site] >> https://spark.apache.org/docs/latest/submitting-applications.html
#	local								Run Spark locally with one worker thread (i.e. no parallelism at all).
#	local[K]							Run Spark locally with K worker threads (ideally, set this to the number of cores on your machine).
#	local[K,F]							Run Spark locally with K worker threads and F maxFailures (see spark.task.maxFailures for an explanation of this variable)
#	local[*]							Run Spark locally with as many worker threads as logical cores on your machine.
#	local[*,F]							Run Spark locally with as many worker threads as logical cores on your machine and F maxFailures.
#	spark://HOST:PORT					Connect to the given Spark standalone cluster master. The port must be whichever one your master is configured to use, which is 7077 by default.
#	spark://HOST1:PORT1,HOST2:PORT2		Connect to the given Spark standalone cluster with standby masters with Zookeeper.
#	mesos://HOST:PORT					Connect to the given Mesos cluster. The port must be whichever one your is configured to use, which is 5050 by default.
#	mesos://zk://HOST:PORT				Connect to the given Mesos cluster using ZooKeeper.
#	yarn								Connect to a YARN cluster in client or cluster mode depending on the value of --deploy-mode. The cluster location will be found based on the HADOOP_CONF_DIR or YARN_CONF_DIR variable.
#	k8s://HOST:PORT						Connect to a Kubernetes cluster in cluster mode.

def basic_configuration():
	spark = SparkSession.builder.appName('basic-configuration').master('local[4]').config('spark.executor.memory', '10g').getOrCreate()
	#conf = SparkConf().set('spark.executor.cores', '3')
	#spark = SparkSession.builder.appName('basic-configuration').master('local').config(conf=conf).enableHiveSupport().getOrCreate()
	#spark = SparkSession.builder.appName('basic-configuration').master('spark://host:7077').config('spark.executor.memory', '10g').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# REF [site] >> https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow
	# Enable Arrow-based columnar data transfers.
	spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

	spark.conf.set('spark.cores.max', '3')
	spark.conf.set('spark.driver.memory','8g')
	spark.conf.set('spark.executor.memory', '8g')
	spark.conf.set('spark.executor.cores', '3')
	spark.conf.set('spark.jars.packages', 'mysql:mysql-connector-java:8.0.12,org.xerial:sqlite-jdbc:3.23.1')

	#print('All configuration =', spark.conf.getAll())  # Error: Not working.
	print('All configuration =', spark.sparkContext.getConf().getAll())

def simple_rdd_example():
	# Create a Spark Context.
	conf = SparkConf().setMaster('local').setAppName('filteringLines')
	sc = SparkContext(conf=conf)
	sc.setLogLevel('WARN')

	nums = sc.parallelize([1, 2, 3, 4])
	squared = nums.map(lambda x: x * x).collect()
	for num in squared:
		print('*****', num)

	# Shut down Spark.
	sc.stop()

def filtering_line_example():
	inputFile = './dataset/README.md'
	outputFile = './filteringlines'

	# Create a Spark Context.
	conf = SparkConf().setMaster('local').setAppName('filteringLines')
	sc = SparkContext(conf=conf)
	sc.setLogLevel('WARN')

	# Load our input data.
	lines = sc.textFile(inputFile)
	print(lines.count())
	print('The first line = {}'.format(lines.first()))

	pythonLines = lines.filter(lambda line: 'Python' in line)
	print('The first line = {}'.format(pythonLines.first()))

	# Save the word count back out to a text file, causing evaluation.
	pythonLines.saveAsTextFile(outputFile)

	# Shut down Spark.
	sc.stop()

def dataframe_basic():
	spark = SparkSession.builder.appName('dataframe-basic').getOrCreate()
	sc = spark.sparkContext
	sc.setLogLevel('WARN')

	if True:
		df = spark.createDataFrame(
			[(123, 'Katie', 19, 'brown'), (234, 'Michael', 22, 'green'), (345, 'Simone', 23, 'blue')],
			('id', 'name', 'age', 'eyeColor')
		)
	elif True:
		stringCSVRDD = sc.parallelize([
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

		#df = spark.read.csv(stringCSVRDD)
		df = spark.createDataFrame(stringCSVRDD, schema)
	elif False:
		df = spark.read.csv('./dataset/swimmers.csv', header=True, infer_schema=True, delimiter=',')
	elif False:
		# Each line is a JSON object.
		df = spark.read.json('./dataset/swimmers.json')
	else:
		#stringJSONRDD = sc.parallelize((
		#	"""{"id": "123", "name": "Katie", "age": 19, "eyeColor": "brown"}""",
		#	"""{"id": "234", "name": "Michael", "age": 22, "eyeColor": "green"}""",
		#	"""{"id": "345", "name": "Simone", "age": 23, "eyeColor": "blue"}"""
		#))
		stringJSONRDD = sc.parallelize((
			{"id": "123", "name": "Katie", "age": 19, "eyeColor": "brown"},
			{"id": "234", "name": "Michael", "age": 22, "eyeColor": "green"},
			{"id": "345", "name": "Simone", "age": 23, "eyeColor": "blue"}
		))

		# Each line is a JSON object.
		df = spark.read.json(stringJSONRDD)

	df.createOrReplaceTempView('swimmers')
	df.printSchema()

	print('df.collect() =', spark.sql('select * from swimmers').collect())
	df.show()
	print('df.take(2) =', df.take(2))
	print('Count =', df.count())

	df.select('id', 'age').filter('age = 22').show()
	df.select(df.id, df.age).filter(22 == df.age).show()
	df.select('name', 'eyeColor').filter('eyeColor like "b%"').show()

def dataframe_operation():
	spark = SparkSession.builder.appName('dataframe-operation').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# Add rows.
	df1 = spark.range(3)
	df2 = spark.range(5)
	df3 = df1.union(df2)
	df3.show()

	# Add columns.
	df1 = spark.createDataFrame([(1, 'a', 23.0), (3, 'B', -23.0)], ('x1', 'x2', 'x3'))
	df2 = df1.withColumn('x4', func.lit(0))
	df2.show()
	df3 = df2.withColumn('x5', func.exp('x3'))
	df3.show(truncate=False)

	df4 = spark.createDataFrame([(1, 'foo'), (2, 'bar')], ('k', 'v'))
	df5 = df3 \
		.join(df4, func.col('x1') == func.col('k'), 'leftouter') \
		.drop('k') \
		.withColumnRenamed('v', 'x6')
	df5.show(truncate=False)

def flight_dataframe_example():
	spark = SparkSession.builder.appName('flight-dataframe-example').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# Set file paths.
	flightPerfFilePath = './dataset/flight/departuredelays.csv'
	airportsFilePath = './dataset/flight/airport-codes-na.txt'

	# Obtain airports dataset.
	airports = spark.read.csv(airportsFilePath, header='true', inferSchema='true', sep='\t')
	airports.createOrReplaceTempView('airports')

	# Obtain departure delays dataset.
	flightPerf = spark.read.csv(flightPerfFilePath, header='true')
	flightPerf.createOrReplaceTempView('FlightPerformance')

	# Cache the departure delays dataset.
	flightPerf.cache()

	# Query sum of flight delays by city and origin code (for Washington state).
	spark.sql("""select a.City, f.origin, sum(f.delay) as Delays
		from FlightPerformance f
			join airports a
				on a.IATA = f.origin where a.State = 'WA'
		group by a.City, f.origin
		order by sum(f.delay) desc"""
	).show()
	# Query sum of flight delays by state (for the US).
	spark.sql("""select a.State, sum(f.delay) as Delays
		from FlightPerformance f
			join airports a
				on a.IATA = f.origin where a.Country = 'USA'
		group by a.State"""
	).show()

def main():
	#basic_configuration()

	#simple_rdd_example()
	#filtering_line_example()

	dataframe_basic()
	dataframe_operation()

	#flight_dataframe_example()

#%%------------------------------------------------------------------

# Usage:
#	python pyspark_basic.py
#	spark-submit pyspark_basic.py
#	spark-submit --master local[4] pyspark_basic.py
#		Run in local mode with 4 cores.
#	spark-submit --master spark://host:7077 --executor-memory 10g pyspark_basic.py
#		Run in a Spark Standalone cluster.

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
