
public class Main {

	public static void main(String[] args) {
		try
		{
			imagej.ImageJ_Main.run(args);  // Not yet implemented.
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
