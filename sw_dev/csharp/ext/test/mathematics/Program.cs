using System;
using System.Collections.Generic;
using System.Text;

namespace mathematics
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Math.NET library ----------------------------------------------------");
                math_net.Math_NET_Main.run(args);

                Console.WriteLine("\nILNumerics library --------------------------------------------------");
                ilnumerics.ILNumerics_Main.run(args);  // not yet implemented.
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
