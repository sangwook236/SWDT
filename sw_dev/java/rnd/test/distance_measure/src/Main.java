
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			fastdtw.FastDTWMain.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
