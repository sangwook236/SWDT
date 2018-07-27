//package com.sangwook.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.commons.lang3.StringUtils;
import java.util.Arrays;

// Usage:
//	REF [doc] >> spark_usage_guide.txt
//
//	When using Maven:
//		mvn clean && mvn compile && mvn package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "Spark_Main" --master local[4] target/spark-example-1.0.0.jar

public class Spark_Main {

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		runSimpleExample();

		Spark_WordCountExample.run(args);
	}

	static void runSimpleExample()
	{
		// Create a Java Spark Context.
		SparkConf conf = new SparkConf().setAppName("simpleExample");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4));
		JavaRDD<Integer> result = rdd.map(new Function<Integer, Integer>() {
			public Integer call(Integer x) { return x * x; }
		});
		System.out.println(StringUtils.join(result.collect(), " ***** "));

		sc.stop();
	}

}
