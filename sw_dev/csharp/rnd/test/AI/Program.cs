using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("AForge.NET library --------------------------------------------------");
                aforge_net.AForge_NET_Main.run(args);  // not yet implemented.
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
