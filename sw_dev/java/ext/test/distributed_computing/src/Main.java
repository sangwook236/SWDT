
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//hadoop.Hadoop_Main.run(args);  // not yet implemented.
			spark.Spark_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
