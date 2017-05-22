
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			fastdtw.FastDTW_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
