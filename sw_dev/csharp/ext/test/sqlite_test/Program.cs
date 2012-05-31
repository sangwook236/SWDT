using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite_test
{
    using System.Data.SQLite;

    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("******************* basic operation");
                BasicOperation.run();
            }
            catch (SQLiteException e)
            {
                Console.WriteLine("System.Data.SQLite.SQLiteException occurred: {0}", e);
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
