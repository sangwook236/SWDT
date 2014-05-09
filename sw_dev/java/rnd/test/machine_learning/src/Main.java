
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("JBoost library ------------------------------------------------------");			
			//jboost.JBoost_Main.run(args);  // not yet implemented.
			
			System.out.println("Weka library --------------------------------------------------------");			
			//weka.Weka_Main.run(args);
			System.out.println("RapidMiner library --------------------------------------------------");			
			rapidminer.RapidMiner_Main.run(args);
			//mahout.Mahout_Main.run(args);  // not yet implemented.
			System.out.println("Mahout library ------------------------------------------------------");			
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
