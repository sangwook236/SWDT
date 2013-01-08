
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			log4j.Log4jMain.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
