package com.sangwook;

import com.sangwook.*;

// Usage:
//	REF [doc] >> spark_usage_guide.txt
//
//	When using Maven:
//		mvn clean && mvn compile && mvn package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "com.sangwook.Main" --master local[4] target/simple-example-1.0.0.jar

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("Spark ---------------------------------------------------------------");
			com.sangwook.spark.Spark_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
