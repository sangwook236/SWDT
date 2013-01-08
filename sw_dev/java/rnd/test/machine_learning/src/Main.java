
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//jboost.JBoostMain.run(args);  // not yet implemented
			
			weka.WekaMain.run(args);
			//rapidminer.RapidMinerMain.run(args);  // not yet implemented
			//mahout.MahoutMain.run(args);  // not yet implemented
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
