﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computing.accord_net
{
    using Accord.Statistics.Distributions.Univariate;
    using Accord.Statistics.Models.Markov.Topology;
    using Accord.Statistics.Models.Markov.Learning;
    using Accord.Statistics.Models.Markov;

    class HiddenMarkovModelExample
    {
        // [ref] http://www.codeproject.com/Articles/541428/Sequence-Classifiers-in-Csharp-Part-I-Hidden-Marko.

        public static void run(string[] args)
        {
            runDiscreteDensityHiddenMarkovModelExample();
            runArbitraryDensityHiddenMarkovModelExample();

            runDiscreteDensityHiddenMarkovModelLearningExample();
            runArbitraryDensityHiddenMarkovModelLearningExample();
            runDiscreteDensityHiddenMarkovClassifierLearningExample();
        }

        static void runDiscreteDensityHiddenMarkovModelExample()
        {
            // Create the transition matrix A.
            double[,] transition =
            {
                { 0.7, 0.3 },
                { 0.4, 0.6 }
            };

            // Create the emission matrix B.
            double[,] emission =
            {
                { 0.1, 0.4, 0.5 },
                { 0.6, 0.3, 0.1 }
            };

            // Create the initial probabilities pi.
            double[] initial =
            {
                0.6, 0.4
            };

            // Create a new hidden Markov model.
            HiddenMarkovModel hmm = new HiddenMarkovModel(transition, emission, initial);

            // Query the probability of a sequence occurring.
            int[] sequence = new int[] { 0, 1, 2 };

            // Evaluate its likelihood.
            double logLikelihood = hmm.Evaluate(sequence);

            // The log-likelihood of the sequence occurring within the model is -3.3928721329161653. 
            Console.WriteLine("log-likelihood = {0}", logLikelihood);

            // Get the Viterbi path of the sequence.
            int[] path = hmm.Decode(sequence, out logLikelihood);

            // The state path will be 1-0-0 and the log-likelihood will be -4.3095199438871337.
            Console.Write("log-likelihood = {0}, Viterbi path = [", logLikelihood);
            foreach (int state in path)
                Console.Write("{0},", state);
            Console.WriteLine("]");
        }

        static void runArbitraryDensityHiddenMarkovModelExample()
        {
            // Create the transition matrix A.
            double[,] transitions = 
            {  
                { 0.7, 0.3 },
                { 0.4, 0.6 }
            };

            // Create the vector of emission densities B.
            GeneralDiscreteDistribution[] emissions = 
            {  
                new GeneralDiscreteDistribution(0.1, 0.4, 0.5),
                new GeneralDiscreteDistribution(0.6, 0.3, 0.1)
            };

            // Create the initial probabilities pi.
            double[] initial =
            {
                0.6, 0.4
            };

            // Create a new hidden Markov model with discrete probabilities.
            var hmm = new HiddenMarkovModel<GeneralDiscreteDistribution>(transitions, emissions, initial);

            // Query the probability of a sequence occurring. We will consider the sequence.
            double[] sequence = new double[] { 0, 1, 2 };

            // Evaluate its likelihood.
            double logLikelihood = hmm.Evaluate(sequence);
            // The log-likelihood of the sequence occurring within the model is -3.3928721329161653.
            Console.WriteLine("log-likelihood = {0}", logLikelihood);

            // Get the Viterbi path of the sequence.
            int[] path = hmm.Decode(sequence, out logLikelihood);

            // The state path will be 1-0-0 and the log-likelihood will be -4.3095199438871337.
            Console.Write("log-likelihood = {0}, Viterbi path = [", logLikelihood);
            foreach (int state in path)
                Console.Write("{0},", state);
            Console.WriteLine("]");
        }

        static void runDiscreteDensityHiddenMarkovModelLearningExample()
        {
            int[][] observationSequences =
            {
                new[] { 0, 1, 2, 3 },
                new[] { 0, 0, 0, 1, 1, 2, 2, 3, 3 },
                new[] { 0, 0, 1, 2, 2, 2, 3, 3 },
                new[] { 0, 1, 2, 3, 3, 3, 3 },
            };

            {
                // Create a hidden Markov model with arbitrary probabilities.
                HiddenMarkovModel hmm = new HiddenMarkovModel(states: 4, symbols: 4);

                // Create a Baum-Welch learning algorithm to teach it.
                BaumWelchLearning trainer = new BaumWelchLearning(hmm);

                // Call its Run method to start learning.
                double averageLogLikelihood = trainer.Run(observationSequences);
                Console.WriteLine("average log-likelihood for the observations = {0}", averageLogLikelihood);

                // Check the probability of some sequences.
                double logLik1 = hmm.Evaluate(new[] { 0, 1, 2, 3 });  // 0.013294354967987107.
                Console.WriteLine("probability = {0}", Math.Exp(logLik1));
                double logLik2 = hmm.Evaluate(new[] { 0, 0, 1, 2, 2, 3 });  // 0.002261813011419950.
                Console.WriteLine("probability = {0}", Math.Exp(logLik2));
                double logLik3 = hmm.Evaluate(new[] { 0, 0, 1, 2, 3, 3 });  // 0.002908045300397080.
                Console.WriteLine("probability = {0}", Math.Exp(logLik3));

                // Violate the form of the training set.
                double logLik4 = hmm.Evaluate(new[] { 3, 2, 1, 0 });  // 0.000000000000000000.
                Console.WriteLine("probability = {0}", Math.Exp(logLik4));
                double logLik5 = hmm.Evaluate(new[] { 0, 0, 1, 3, 1, 1 });  // 0.000000000113151816.
                Console.WriteLine("probability = {0}", Math.Exp(logLik5));
            }

            {
                // Create a hidden Markov model with arbitrary probabilities.
                var hmm = new HiddenMarkovModel<GeneralDiscreteDistribution>(states: 4, emissions: new GeneralDiscreteDistribution(symbols: 4));

                // Create a Baum-Welch learning algorithm to teach it
                // until the difference in the average log-likelihood changes only by as little as 0.0001
                // and the number of iterations is less than 1000.
                var trainer = new BaumWelchLearning<GeneralDiscreteDistribution>(hmm)
                {
                    Tolerance = 0.0001,
                    Iterations = 1000,
                };

                // Call its Run method to start learning.
                double averageLogLikelihood = trainer.Run(observationSequences);
                Console.WriteLine("average log-likelihood for the observations = {0}", averageLogLikelihood);

                // Check the probability of some sequences.
                double logLik1 = hmm.Evaluate(new[] { 0, 1, 2, 3 });  // 0.013294354967987107.
                Console.WriteLine("probability = {0}", Math.Exp(logLik1));
                double logLik2 = hmm.Evaluate(new[] { 0, 0, 1, 2, 2, 3 });  // 0.002261813011419950.
                Console.WriteLine("probability = {0}", Math.Exp(logLik2));
                double logLik3 = hmm.Evaluate(new[] { 0, 0, 1, 2, 3, 3 });  // 0.002908045300397080.
                Console.WriteLine("probability = {0}", Math.Exp(logLik3));

                // Violate the form of the training set.
                double logLik4 = hmm.Evaluate(new[] { 3, 2, 1, 0 });  // 0.000000000000000000.
                Console.WriteLine("probability = {0}", Math.Exp(logLik4));
                double logLik5 = hmm.Evaluate(new[] { 0, 0, 1, 3, 1, 1 });  // 0.000000000113151816.
                Console.WriteLine("probability = {0}", Math.Exp(logLik5));
            }
        }

        static void runArbitraryDensityHiddenMarkovModelLearningExample()
        {
            // Create continuous sequences.
            //  In the sequences below, there seems to be two states, one for values between 0 and 1 and another for values between 5 and 7.
            //  The states seems to be switched on every observation.
            double[][] observationSequences = new double[][] 
            {
                new double[] { 0.1, 5.2, 0.3, 6.7, 0.1, 6.0 },
                new double[] { 0.2, 6.2, 0.3, 6.3, 0.1, 5.0 },
                new double[] { 0.1, 7.0, 0.1, 7.0, 0.2, 5.6 },
            };

            // Creates a continuous hidden Markov Model with two states organized in a ergoric topology
            // and an underlying univariate Normal distribution as probability density. 
            var hmm = new HiddenMarkovModel<NormalDistribution>(topology: new Ergodic(states: 2), emissions: new NormalDistribution());

            // Configure the learning algorithms to train the sequence classifier
            // until the difference in the average log-likelihood changes only by as little as 0.0001.
            var trainer = new BaumWelchLearning<NormalDistribution>(hmm)
            {
                Tolerance = 0.0001,
                Iterations = 0,
            };

            // Fit the model.
            double averageLogLikelihood = trainer.Run(observationSequences);
            Console.WriteLine("average log-likelihood for the observations = {0}", averageLogLikelihood);

            // The log-probability of the sequences learned.
            double logLik1 = hmm.Evaluate(new[] { 0.1, 5.2, 0.3, 6.7, 0.1, 6.0 });  // -0.12799388666109757.
            double logLik2 = hmm.Evaluate(new[] { 0.2, 6.2, 0.3, 6.3, 0.1, 5.0 });  // 0.01171157434400194.

            // The log-probability of an unrelated sequence.
            double logLik3 = hmm.Evaluate(new[] { 1.1, 2.2, 1.3, 3.2, 4.2, 1.0 });  // -298.7465244473417.

            // Transform the log-probabilities to actual probabilities.
            Console.WriteLine("probability = {0}", Math.Exp(logLik1));  // 0.879.
            Console.WriteLine("probability = {0}", Math.Exp(logLik2));  // 1.011.
            Console.WriteLine("probability = {0}", Math.Exp(logLik3));  // 0.000.

            // Ask the model to decode one of the sequences.
            // The state variable will contain: { 0, 1, 0, 1, 0, 1 }.
            double logLikelihood = 0.0;
            int[] path = hmm.Decode(new[] { 0.1, 5.2, 0.3, 6.7, 0.1, 6.0 }, out logLikelihood);
            Console.Write("log-likelihood = {0}, Viterbi path = [", logLikelihood);
            foreach (int state in path)
                Console.Write("{0},", state);
            Console.WriteLine("]");
        }

        static void runDiscreteDensityHiddenMarkovClassifierLearningExample()
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

            // Use a single topology for all inner models.
            ITopology forward = new Forward(states: 3);

            // Create a hidden Markov classifier with the given topology.
            HiddenMarkovClassifier hmc = new HiddenMarkovClassifier(classes: 3, topology: forward, symbols: 4);

            // Create a algorithms to teach each of the inner models.
            var trainer = new HiddenMarkovClassifierLearning(
                hmc,
                // Specify individual training options for each inner model.
                modelIndex => new BaumWelchLearning(hmc.Models[modelIndex])
                {
                    Tolerance = 0.001,  // iterate until log-likelihood changes less than 0.001.
                    Iterations = 0  // don't place an upper limit on the number of iterations.
                }
            );

            // Call its Run method to start learning.
            double averageLogLikelihood = trainer.Run(observationSequences, classLabels);
            Console.WriteLine("average log-likelihood for the observations = {0}", averageLogLikelihood);

            // Check the output classificaton label for some sequences. 
            int y1 = hmc.Compute(new[] { 0, 1, 1, 1, 0 });  // output is y1 = 0.
            Console.WriteLine("output class = {0}", y1);
            int y2 = hmc.Compute(new[] { 0, 0, 1, 1, 0, 0 });  // output is y2 = 0.
            Console.WriteLine("output class = {0}", y2);

            int y3 = hmc.Compute(new[] { 2, 2, 2, 2, 1, 1 });  // output is y3 = 1.
            Console.WriteLine("output class = {0}", y3);
            int y4 = hmc.Compute(new[] { 2, 2, 1, 1 });  // output is y4 = 1.
            Console.WriteLine("output class = {0}", y4);

            int y5 = hmc.Compute(new[] { 0, 0, 1, 3, 3, 3 });  // output is y5 = 2.
            Console.WriteLine("output class = {0}", y4);
            int y6 = hmc.Compute(new[] { 2, 0, 2, 2, 3, 3 });  // output is y6 = 2.
            Console.WriteLine("output class = {0}", y6);
        }
    }
}
