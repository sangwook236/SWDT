#!/usr/bin/env python

from pyspark import SparkConf, SparkContext

def simple_example():
	conf = SparkConf().setMaster('local').setAppName('My App')
	sc = SparkContext(conf=conf)

	lines = sc.textFile('README.md')
	print(lines.count())
	print('The first line = {}'.format(lines.first()))

	pythonLines = lines.filter(lambda line: 'Python' in line)
	print('The first line = {}'.format(pythonLines.first()))

	sc.stop()

def main():
	simple_example()

#%%------------------------------------------------------------------

# Usage:
#	Download winutils.exe
#		http://public-repo-1.hortonworks.com/hdp-win-alpha/winutils.exe
#		https://github.com/steveloughran/winutils
#	Set environment variable.
#		If winutils.exe is in ${WINUTILS_HOME}/bin,
#			set HADOOP_HOME=${WINUTILS_HOME}
#	spark-submit pyspark_basic.py

if '__main__' == __name__:
	main()
