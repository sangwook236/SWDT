
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//collections.CollectionsMain.run(args);  // not yet implemented
			//configuration.ConfigurationMain.run(args);
			//dbutils.DbUtilsMain.run(args);  // not yet implemented
			//lang.LangMain.run(args);  // not yet implemented
			math.MathMain.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
