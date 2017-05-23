package spark;

import org.apache.spark.SparkConf;
//import org.apache.spark.api.java.JavaSparkContext;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;
import java.io.Serializable;
import java.util.Arrays;

public class Spark_Main {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		// FIXME [modify] >> Don't know how to run in Eclipse.

		String inputFile = "./SimpleApp.java";
		String outputFile = "./wordcount";

		SparkConf conf = new SparkConf().setAppName("WordCountApp");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<String> input = sc.textFile(inputFile);

		JavaRDD<String> words = input.flatMap(
			new FlatMapFunction<String, String>()
			{
				public Iterable<String> call(String x)
				{
					return Arrays.asList(x.split(" "));
				}
			}
		);

		JavaPairRDD<String, Integer> counts = words.mapToPair(
			new PairFunction<String, String, Integer>()
			{
				public Tuple2<String, Integer> call(String x)
				{
					return new Tuple2(x, 1);
				}
			}
		).reduceByKey(
			new Function2<Integer, Integer, Integer>()
			{
				public Integer call(Integer x, Integer y)  { return x + y; }
			}
		);

		counts.saveAsTextFile(outputFile);
	}

}
