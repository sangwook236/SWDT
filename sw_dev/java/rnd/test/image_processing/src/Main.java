
public class Main {

	public static void main(String[] args) {
		try
		{
			System.out.println("ImageJ library ------------------------------------------------------");			
			imagej.ImageJ_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
