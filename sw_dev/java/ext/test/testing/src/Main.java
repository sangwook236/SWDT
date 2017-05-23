
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("JUnit ---------------------------------------------------------------");
			junit.JUnit_Main.run(args);

			System.out.println("Hamcrest library ----------------------------------------------------");
			//hamcrest.Hamcrest_Main.run(args);  // Not yet implemented.
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
