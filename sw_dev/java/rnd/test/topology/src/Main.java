
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			fastdtw.FastDTW_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
