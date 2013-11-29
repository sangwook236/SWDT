
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			//jboost.JBoost_Main.run(args);  // not yet implemented.
			
			weka.Weka_Main.run(args);
			//rapidminer.RapidMiner_Main.run(args);  // not yet implemented.
			//mahout.Mahout_Main.run(args);  // not yet implemented.
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
