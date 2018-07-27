//package com.sangwook.spark;

import org.apache.spark.SparkConf;
//import org.apache.spark.api.java.JavaSparkContext;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;
import java.io.Serializable;
import java.util.Arrays;

public class Spark_WordCountExample {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		// FIXME [modify] >> Don't know how to run in Eclipse.

		String inputFile = "./README.md";
		String outputFile = "./wordcount";

		// Create a Java Spark Context.
		SparkConf conf = new SparkConf().setAppName("wordCount");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load our input data.
		JavaRDD<String> input = sc.textFile(inputFile);

		// Split up into words.
		JavaRDD<String> words = input.flatMap(l -> Arrays.asList(l.split(" ")).iterator());

		// Transform into pairs and count.
		JavaPairRDD<String, Integer> counts = words.mapToPair(
			new PairFunction<String, String, Integer>() {
				public Tuple2<String, Integer> call(String x) {
					return new Tuple2(x, 1);
				}
			}
		).reduceByKey(new Function2<Integer, Integer, Integer>() {
			public Integer call(Integer x, Integer y){ return x + y; }
		});

		// Save the word count back out to a text file, causing evaluation.
		counts.saveAsTextFile(outputFile);

		sc.stop();
	}

}
