using System;
using System.Collections.Generic;
using System.Text;

namespace testing
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("nunit library -------------------------------------------------------");

                throw new NotImplementedException();
            }
            catch (Exception ex)
            {
                Console.WriteLine("System.Exception occurred: {0}", ex);
            }

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey();
        }
    }
}
