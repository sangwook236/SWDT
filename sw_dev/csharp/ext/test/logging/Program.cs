using System;
using System.Collections.Generic;
using System.Text;

namespace logging
{
    class Program
    {
        static void Main(string[] args) 
        {
            try
            {
                Console.WriteLine("log4net library -----------------------------------------------------");
                log4net.log4net_Main.run(args);
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
