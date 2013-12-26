using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace serialization
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Jayrock library -----------------------------------------------------");
                jayrock.Jayrock_Main.run(args);
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
