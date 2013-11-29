
public class Main {

	public static void main(String[] args) {
		try
		{
			imagej.ImageJ_Main.run(args);  // not yet implemented.
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
