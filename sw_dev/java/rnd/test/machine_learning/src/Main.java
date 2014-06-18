
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

			System.out.println("Mahout library ------------------------------------------------------");			
			//mahout.Mahout_Main.run(args);  // not yet implemented.

			System.out.println("Encog Machine Learning Framework ------------------------------------");
			//	-. Java, .NET and C/C++.
			//	-. neural network.
			//		ADALINE neural network.
			//		adaptive resonance theory 1 (ART1).
			//		bidirectional associative memory (BAM).
			//		Boltzmann machine.
			//		feedforward neural network.
			//		recurrent neural network.
			//		Hopfield neural network.
			//		radial basis function network (RBFN).
			//		neuroevolution of augmenting topologies (NEAT).
			//		(recurrent) self organizing map (SOM).
			//encog.Encog_Main.run(args);  // not yet implemented.
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
