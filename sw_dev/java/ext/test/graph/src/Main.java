
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("Hadoop --------------------------------------------------------------");
			System.out.println("GraphChi ------------------------------------------------------------");
			graphchi.GraphChi_Main.run(args);  // Not yet implemented.
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
