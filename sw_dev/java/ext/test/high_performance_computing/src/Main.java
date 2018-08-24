
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("Hadoop --------------------------------------------------------------");
			//hadoop.Hadoop_Main.run(args);  // Not yet implemented.

			System.out.println("Spark ---------------------------------------------------------------");
			spark.Spark_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
