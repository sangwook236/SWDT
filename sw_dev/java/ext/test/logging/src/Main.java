
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			log4j.Log4j_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
