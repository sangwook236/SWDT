
public class Main {

	public static void main(String[] args) {
		try
		{
			imagej.ImageJMain.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
