
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("Swing ---------------------------------------------------------------");
			//swing.Swing_Main.run(args);  // Not yet implemented.

			System.out.println("SWT -----------------------------------------------------------------");
			//swt.SWT_Main.run(args);  // Not yet implemented.

			System.out.println("Apache Pivo ---------------------------------------------------------");
			apache_pivot.ApachePivot_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
