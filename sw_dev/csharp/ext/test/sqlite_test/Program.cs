using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite_test
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("******************* basic operation");
                BasicOperation.run();
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
