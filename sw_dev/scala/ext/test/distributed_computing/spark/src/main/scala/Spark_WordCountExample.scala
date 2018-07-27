//package com.sangwook.spark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object Spark_WordCountExample {
	def run(args: Array[String])
	{
		// Create a Scala Spark Context.
		val conf = new SparkConf().setMaster("local").setAppName("wordCount")
		val sc = new SparkContext(conf)

		var inputFile = "./README.md"
		var outputFile = "./wordcount"

		// Load our input data.
		val input = sc.textFile(inputFile)

		// Split it up into words.
		val words = input.flatMap(line => line.split(" "))

		// Transform into pairs and count.
		val counts = words.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}

		// Save the word count back out to a text file, causing evaluation.
		counts.saveAsTextFile(outputFile)

		sc.stop()
	}
}
