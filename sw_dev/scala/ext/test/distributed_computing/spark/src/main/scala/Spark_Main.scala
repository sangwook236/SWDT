//package com.sangwook.spark

// Usage:
//	REF [doc] >> spark_usage_guide.txt
//
//	When using sbt:
//		sbt clean package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "Spark_Main" --master local[4] target/scala-2.11/spark-example_2.11-1.0.0.jar
//	When using Maven:
//		mvn clean && mvn compile && mvn package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "Spark_Main" --master local[4] target/spark-example-1.0.0.jar

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object Spark_Main {
	def main(args: Array[String])
	{
		runSimpleExample()

		Spark_WordCountExample.run(args)
	}

	def runSimpleExample()
	{
		// Create a Scala Spark Context.
		val conf = new SparkConf().setMaster("local").setAppName("wordCount")
		val sc = new SparkContext(conf)

		val input = sc.parallelize(List(1, 2, 3, 4))
		val result = input.map(x => x * x)
		println(result.collect().mkString(" ***** "))

		sc.stop()
	}
}
