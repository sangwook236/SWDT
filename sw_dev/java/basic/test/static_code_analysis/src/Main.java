public class Main {

	public static void main(String[] args) {
		try
		{
			pmd.PMD_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
