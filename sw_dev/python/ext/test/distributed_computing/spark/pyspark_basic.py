#!/usr/bin/env python

from pyspark import SparkConf, SparkContext

def simple_example():
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
	inputFile = './README.md'
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

def main():
	simple_example()
	filtering_line_example()

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
