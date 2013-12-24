using System;
using System.Collections.Generic;
using System.Text;

namespace math_library
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Math.NET library ----------------------------------------------------");
                Math_NET_Main.run(args);
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
