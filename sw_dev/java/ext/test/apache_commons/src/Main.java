
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//collections.Collections_Main.run(args);  // Not yet implemented.
			//configuration.Configuration_Main.run(args);
			//dbutils.DbUtils_Main.run(args);  // Not yet implemented.
			//lang.Lang_Main.run(args);  // Not yet implemented.
			math.Math_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
