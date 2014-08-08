using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace machine_learning
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("numl library --------------------------------------------------------");
                numl.numl_Main.run(args);  // not yet implemented.

                Console.WriteLine("Encog Machine Learning Framework ------------------------------------");
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
                encog.Encog_Main.run(args);  // not yet implemented.
            }
            catch (Exception e)
            {
                Console.WriteLine("System.Exception occurred: {0}", e);
            }

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey();
        }
    }
}
