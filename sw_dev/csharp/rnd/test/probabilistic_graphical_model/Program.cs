using System;
using System.Collections.Generic;
using System.Text;

namespace probabilistic_graphical_model
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Infer.NET library ---------------------------------------------------");
                Infer_NET_Main.run(args);  // not yet implemented.
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
