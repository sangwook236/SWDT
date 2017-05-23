
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("SQLite --------------------------------------------------------------");
			sqlite.SQLite_Main.run(args);
			
			System.out.println("MySQL ---------------------------------------------------------------");
			//mysql.MySQLMain.run(args);
		}
		catch (ClassNotFoundException ex)
		{
    		System.err.println(ex);
		}
	}

}
