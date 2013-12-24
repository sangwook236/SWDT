using System;
using System.Collections.Generic;
using System.Text;

namespace database
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("sqlite --------------------------------------------------------------");
                sqlite.sqlite_Main.run(args);
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
