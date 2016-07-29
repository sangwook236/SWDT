
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			junit.JUnit_Main.run(args);
			
			//hamcrest.Hamcrest_Main.run(args);  // not yet implemented.
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
