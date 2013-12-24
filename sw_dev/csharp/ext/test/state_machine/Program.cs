using System;
using System.Collections.Generic;
using System.Text;

namespace state_machine
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                throw new NotImplementedException();
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
