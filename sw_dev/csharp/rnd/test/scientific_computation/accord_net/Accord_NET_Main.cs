using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computation.accord_net
{
    class Accord_NET_Main
    {
        public static void run(string[] args)
        {
            Console.WriteLine("Hidden Markov model (HMM) -------------------------------------------");
            HiddenMarkovModel.run(args);
        }
    }
}
