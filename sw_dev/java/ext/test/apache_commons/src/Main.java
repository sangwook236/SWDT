
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//collections.Collections_Main.run(args);  // not yet implemented.
			//configuration.Configuration_Main.run(args);
			//dbutils.DbUtils_Main.run(args);  // not yet implemented.
			//lang.Lang_Main.run(args);  // not yet implemented.
			math.Math_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
