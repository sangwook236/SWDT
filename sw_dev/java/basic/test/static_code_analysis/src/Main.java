public class Main {

	public static void main(String[] args) {
		try
		{
			pmd.PMD_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
