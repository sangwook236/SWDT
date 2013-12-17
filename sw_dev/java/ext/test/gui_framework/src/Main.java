
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//swing.SwingMain.run(args);  // not yet implemented
			//swt.SWTMain.run(args);  // not yet implemented
	
			apache_pivot.ApachePivot_Main.run(args);
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
