#!/usr/bin/env spark-submit

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *

def simple_rdd_example():
	# Create a Spark Context.
	conf = SparkConf().setMaster('local').setAppName('filteringLines')
	sc = SparkContext(conf=conf)

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

def simple_dataframe_example():
	spark = SparkSession.builder.appName('simple-dataframe-example').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()
	sc = spark.sparkContext

	if False:
		df = spark.read.csv('./dataset/swimmers.csv', header=True, infer_schema=True, delimiter=',')
	elif True:
		stringCSVRDD = sc.parallelize([
			(123, 'Katie', 19, 'brown'),
			(234, 'Michael', 22, 'green'),
			(345, 'Simone', 23, 'blue')
		])
		# Specify schema.
		schema = StructType([
			StructField('id', LongType(), True),
			StructField('name', StringType(), True),
			StructField('age', LongType(), True),
			StructField('eyeColor', StringType(), True)
		])

		#df = spark.read.csv(stringCSVRDD)
		df = spark.createDataFrame(stringCSVRDD, schema)
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

	spark.sql('select * from swimmers').collect()
	df.show()
	df.take(2)
	df.count()

	df.select('id', 'age').filter('age = 22').show()
	df.select(df.id, df.age).filter(22 == swimmers.age).show()
	df.select('name', 'eyeColor').filter('eyeColor like "b%"').show()

def flight_example():
	spark = SparkSession.builder.appName('flight-example').config('spark.sql.crossJoin.enabled', 'true').getOrCreate()

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
	simple_rdd_example()
	#filtering_line_example()

	simple_dataframe_example()
	flight_example()

#%%------------------------------------------------------------------

# Usage:
#	Download winutils.exe
#		http://public-repo-1.hortonworks.com/hdp-win-alpha/winutils.exe
#		https://github.com/steveloughran/winutils
#	Set environment variable.
#		If winutils.exe is in ${WINUTILS_HOME}/bin,
#			set HADOOP_HOME=${WINUTILS_HOME}
#
#	spark-submit pyspark_basic.py
#	spark-submit --master local[4] pyspark_basic.py
#		Run in local mode with 4 cores.
#	spark-submit --master spark://host:7077 --executor-memory 10g pyspark_basic.py
#		Run in a Spark Standalone cluster.

if '__main__' == __name__:
	main()
