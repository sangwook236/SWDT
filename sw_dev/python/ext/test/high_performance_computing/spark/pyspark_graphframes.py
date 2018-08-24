#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as func
from graphframes import *
import traceback, sys

def flight_example():
	spark = SparkSession.builder.appName('simple-tensorframes-example-1').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	# Set file paths.
	tripdelaysFilePath = 'dataset/flight/departuredelays.csv'
	airportsnaFilePath = 'dataset/flight/airport-codes-na.txt'

	# Obtain airports dataset with a tab-delimited header.
	airportsna = spark.read.csv(airportsnaFilePath, header='true', inferSchema='true', sep='\t')
	airportsna.createOrReplaceTempView('airports_na')
	# Obtain departure Delays data with a comma-delimited header.
	departureDelays = spark.read.csv(tripdelaysFilePath, header='true')
	departureDelays.createOrReplaceTempView('departureDelays')
	departureDelays.cache()

	# Available IATA codes from the departuredelays sample dataset.
	tripIATA = spark.sql('select distinct iata from (select \
		distinct origin as iata from departureDelays union all select \
		distinct destination as iata from departureDelays) a')
	tripIATA.createOrReplaceTempView('tripIATA')

	# Only include airports with at least one trip from the 'departureDelays' dataset.
	airports = spark.sql('select f.IATA, f.City, f.State, f.Country from airports_na f join tripIATA t on t.IATA = f.IATA')
	airports.createOrReplaceTempView('airports')
	airports.cache()

	# Build 'departureDelays_geo' dataframe.
	# Obtain key attributes such as Date of flight, delays, distance, and airport information (Origin, Destination).
	departureDelays_geo = spark.sql("select cast(f.date as int) as \
		tripid, cast(concat(concat(concat(concat(concat(concat('2014-', \
		concat(concat(substr(cast(f.date as string), 1, 2), '-')), \
		substr(cast(f.date as string), 3, 2)), ''), substr(cast(f.date \
		as string), 5, 2)), ':'), substr(cast(f.date as string), 7, \
		2)), ':00') as timestamp) as localdate, cast(f.delay as int), \
		cast(f.distance as int), f.origin as src, f.destination as dst, \
		o.city as city_src, d.city as city_dst, o.state as state_src, \
		d.state as state_dst from departuredelays f join airports o on \
		o.iata = f.origin join airports d on d.iata = f.destination")
	# Create Temporary View and cache.
	departureDelays_geo.createOrReplaceTempView('departureDelays_geo')
	departureDelays_geo.cache()

	# Review the top 10 rows of the 'departureDelays_geo' dataframe.
	departureDelays_geo.show(10)

	# Create vertices (airports) and edges (flights).
	tripVertices = airports.withColumnRenamed('IATA', 'id').distinct()
	tripEdges = departureDelays_geo.select('tripid', 'delay', 'src', 'dst', 'city_dst', 'state_dst')
	# Cache vertices and edges.
	tripEdges.cache()
	tripVertices.cache()

	# Create a GraphFrame.
	tripGraph = GraphFrame(tripVertices, tripEdges)

	# Execute simple queries.
	print('Airports: %d' % tripGraph.vertices.count())
	print('Trips: %d' % tripGraph.edges.count())

	# Determine the longest delay in this dataset.
	tripGraph.edges.groupBy().max('delay').show()

	# Determine the number of delayed versus on-time/early flights.
	print('On-time / Early Flights: %d' % tripGraph.edges.filter('delay <= 0').count())
	print('Delayed Flights: %d' % tripGraph.edges.filter('delay > 0').count())

	# What flights departing Seattle are most likely to have significant delays?
	tripGraph.edges \
		.filter("src = 'SEA' and delay > 0") \
		.groupBy('src', 'dst') \
		.avg('delay') \
		.sort(desc('avg(delay)')) \
		.show(5)

	# What states tend to have significant delays departing from Seattle?
	# States with the longest cumulative delays (with individual delays > 100 minutes) (origin: Seattle).
	tripGraph.edges.filter("src = 'SEA' and delay > 100").show()

	# The degree around a vertex.
	# Ask for the top 20 busiest airports (most flights in and out) from our graph.
	tripGraph.degrees.sort(desc('degree')).limit(20).show()
	# The top 20 inDegrees (that is, incoming flights).
	tripGraph.inDegrees.sort(desc('inDegree')).limit(20).show()
	# The top 20 outDegrees (that is, outgoing flights).
	tripGraph.outDegrees.sort(desc('outDegree')).limit(20).show()

	# Determine the top transfer airports.
	# Calculate the inDeg (flights into the airport) and outDeg (flights leaving the airport).
	inDeg = tripGraph.inDegrees
	outDeg = tripGraph.outDegrees
	# Calculate the degreeRatio (inDeg/outDeg).
	degreeRatio = inDeg.join(outDeg, inDeg.id == outDeg.id) \
		.drop(outDeg.id) \
		.selectExpr('id', 'double(inDegree)/double(outDegree) as degreeRatio') \
		.cache()
	# Join back to the 'airports' dataframe (instead of registering temp table as above).
	transferAirports = degreeRatio.join(airports, degreeRatio.id == airports.IATA) \
		.selectExpr('id', 'city', 'degreeRatio') \
		.filter('degreeRatio between 0.9 and 1.1')
	# List out the top 10 transfer city airports.
	transferAirports.orderBy('degreeRatio').limit(10).show()

	# Generate motifs.
	motifs = tripGraph.find('(a)-[ab]->(b); (b)-[bc]->(c)') \
		.filter("(b.id = 'SFO') and (ab.delay > 500 or bc.delay > 500) and bc.tripid > ab.tripid and bc.tripid < ab.tripid + 10000")
	# Display motifs.
	motifs.show()

	# Determine airport ranking of importance using 'pageRank'.
	ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
	# Display the pageRank output.
	ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(20).show()

	# Determine the most popular non-stop flights.
	topTrips = tripGraph.edges.groupBy('src', 'dst').agg(func.count('delay').alias('trips'))
	# Show the top 20 most popular flights (single city hops).
	topTrips.orderBy(topTrips.trips.desc()).limit(20).show()

	# Obtain list of direct flights between SEA and SFO.
	filteredPaths = tripGraph.bfs(fromExpr = "id = 'SEA'", toExpr = "id = 'SFO'", maxPathLength = 1)
	# Display list of direct flights.
	filteredPaths.show()

	# Obtain list of direct flights between SFO and BUF.
	filteredPaths = tripGraph.bfs(fromExpr = "id = 'SFO'", toExpr = "id = 'BUF'", maxPathLength = 1)
	# Display list of direct flights.
	filteredPaths.show()

	# Obtain list of one-stop flights between SFO and BUF.
	filteredPaths = tripGraph.bfs(fromExpr = "id = 'SFO'", toExpr = "id = 'BUF'", maxPathLength = 2)
	# Display list of flights.
	filteredPaths.show()

	# Display most popular layover cities by descending count.
	filteredPaths.groupBy('v1.id', 'v1.City').count().orderBy(desc('count')).limit(10).show()

def main():
	flight_example()

#%%------------------------------------------------------------------

# Usage:
#	python pyspark_graphframes.py
#	spark-submit --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark_graphframes.py
#	spark-submit --master local[4] --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark_graphframes.py
#	spark-submit --master spark://host:7077 --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 --executor-memory 10g pyspark_graphframes.py

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
