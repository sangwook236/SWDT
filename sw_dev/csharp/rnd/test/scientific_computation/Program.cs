using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace scientific_computation
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("accord.NET library --------------------------------------------------");
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
