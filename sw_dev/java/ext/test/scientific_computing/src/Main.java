
public class Main {

	public static void main(String[] args) {
		try
		{
			System.out.println("COLT library --------------------------------------------------------");
			colt.Colt_Main.run(args);  // Not yet implemented.
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
