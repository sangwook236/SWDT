using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computing
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("AForge.NET library --------------------------------------------------");
                //aforge_net.AForge_NET_Main.run(args);  // not yet implemented.

                Console.WriteLine("Accord.NET library --------------------------------------------------");
                accord_net.Accord_NET_Main.run(args);  // not yet implemented.
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
