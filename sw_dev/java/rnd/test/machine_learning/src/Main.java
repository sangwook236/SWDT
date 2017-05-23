
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("JBoost library ------------------------------------------------------");			
			//jboost.JBoost_Main.run(args);  // Not yet implemented.
			
			System.out.println("Weka library --------------------------------------------------------");			
			//weka.Weka_Main.run(args);

			System.out.println("RapidMiner library --------------------------------------------------");			
			rapidminer.RapidMiner_Main.run(args);

			System.out.println("Mahout library ------------------------------------------------------");			
			//mahout.Mahout_Main.run(args);  // Not yet implemented.

			System.out.println("Encog Machine Learning Framework ------------------------------------");
			//	- Java, .NET and C/C++.
			//	- Neural network.
			//		ADALINE neural network.
			//		Adaptive resonance theory 1 (ART1).
			//		Bidirectional associative memory (BAM).
			//		Boltzmann machine.
			//		Feedforward neural network.
			//		Recurrent neural network (rnn).
			//		Hopfield neural network.
			//		Radial basis function network (RBFN).
			//		Neuroevolution of augmenting topologies (NEAT).
			//		(Recurrent) self organizing map (SOM).
			//encog.Encog_Main.run(args);  // Not yet implemented.
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
