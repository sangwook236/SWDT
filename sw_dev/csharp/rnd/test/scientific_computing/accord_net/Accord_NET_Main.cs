using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computing.accord_net
{
    class Accord_NET_Main
    {
        public static void run(string[] args)
        {
            Console.WriteLine("Hidden Markov Model (HMM) -------------------------------------------");
            HiddenMarkovModelExample.run(args);

            Console.WriteLine("Conditional Random Field (CRF) --------------------------------------");
            //ConditionalRandomFieldExample.run(args);
            Console.WriteLine("Hidden Conditional Random Field (HCRF) ------------------------------");
            HiddenConditionalRandomFieldExample.run(args);
        }
    }
}
