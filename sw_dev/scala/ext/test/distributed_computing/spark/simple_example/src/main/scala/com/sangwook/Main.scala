package com.sangwook

// Usage:
//	REF [doc] >> spark_usage_guide.txt
//
//	When using sbt:
//		sbt clean package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "com.sangwook.Main" --master local[4] target/scala-2.11/simple-example_2.11-1.0.0.jar
//	When using Maven:
//		mvn clean && mvn compile && mvn package
//		${SPARK_HOME}/bin/spark-submit2.cmd --class "com.sangwook.Main" --master local[4] target/simple-example-1.0.0.jar

object Main {
	def main(args: Array[String])
	{
		spark.Spark_Main.run(args);
	}
}
