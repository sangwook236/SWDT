using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computing.accord_net
{
    using Accord.Statistics.Models.Fields.Functions;
    using Accord.Statistics.Models.Fields.Learning;
    using Accord.Statistics.Models.Fields;

    class HiddenConditionalRandomFieldExample
    {
        // [ref] http://www.codeproject.com/Articles/559535/Sequence-Classifiers-in-Csharp-Part-II-Hidden-Cond.

        public static void run(string[] args)
        {
            runHiddenConditionalRandomFieldLearningExample();
        }

        static void runHiddenConditionalRandomFieldLearningExample()
        {
            // Observation sequences should only contain symbols that are greater than or equal to 0, and lesser than the number of symbols.
            int[][] observationSequences =
            {
                // First class of sequences: starts and ends with zeros, ones in the middle.
                new[] { 0, 1, 1, 1, 0 },
                new[] { 0, 0, 1, 1, 0, 0 },
                new[] { 0, 1, 1, 1, 1, 0 },
 
                // Second class of sequences: starts with twos and switches to ones until the end.
                new[] { 2, 2, 2, 2, 1, 1, 1, 1, 1 },
                new[] { 2, 2, 1, 2, 1, 1, 1, 1, 1 },
                new[] { 2, 2, 2, 2, 2, 1, 1, 1, 1 },
 
                // Third class of sequences: can start with any symbols, but ends with three.
                new[] { 0, 0, 1, 1, 3, 3, 3, 3 },
                new[] { 0, 0, 0, 3, 3, 3, 3 },
                new[] { 1, 0, 1, 2, 2, 2, 3, 3 },
                new[] { 1, 1, 2, 3, 3, 3, 3 },
                new[] { 0, 0, 1, 1, 3, 3, 3, 3 },
                new[] { 2, 2, 0, 3, 3, 3, 3 },
                new[] { 1, 0, 1, 2, 3, 3, 3, 3 },
                new[] { 1, 1, 2, 3, 3, 3, 3 },
            };

            // Consider their respective class labels.
            // Class labels have to be zero-based and successive integers.
            int[] classLabels =
            {
                0, 0, 0,  // Sequences 1-3 are from class 0.
                1, 1, 1,  // Sequences 4-6 are from class 1.
                2, 2, 2, 2, 2, 2, 2, 2  // Sequences 7-14 are from class 2.
            };

            // Create the Hidden Conditional Random Field using a set of discrete features.
            var function = new MarkovDiscreteFunction(states: 3, symbols: 4, outputClasses: 3);
            var hcrf = new HiddenConditionalRandomField<int>(function);

            // Create a learning algorithm.
            var trainer = new HiddenResilientGradientLearning<int>(hcrf)
            {
                Iterations = 50
            };

            // Run the algorithm and learn the models.
            double error = trainer.Run(observationSequences, classLabels);
            Console.WriteLine("the error in the last iteration = {0}", error);

            // Check the output classificaton label for some sequences.
            int y1 = hcrf.Compute(new[] { 0, 1, 1, 1, 0 });  // output is y1 = 0.
            Console.WriteLine("output class = {0}", y1);
            int y2 = hcrf.Compute(new[] { 0, 0, 1, 1, 0, 0 });  // output is y2 = 0.
            Console.WriteLine("output class = {0}", y2);

            int y3 = hcrf.Compute(new[] { 2, 2, 2, 2, 1, 1 });  // output is y3 = 1.
            Console.WriteLine("output class = {0}", y3);
            int y4 = hcrf.Compute(new[] { 2, 2, 1, 1 });  // output is y4 = 1.
            Console.WriteLine("output class = {0}", y4);

            int y5 = hcrf.Compute(new[] { 0, 0, 1, 3, 3, 3 });  // output is y5 = 2.
            Console.WriteLine("output class = {0}", y5);
            int y6 = hcrf.Compute(new[] { 2, 0, 2, 2, 3, 3 });  // output is y6 = 2.
            Console.WriteLine("output class = {0}", y6);
        }
    }
}
