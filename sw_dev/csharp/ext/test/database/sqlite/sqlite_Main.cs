using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace database
{
    using System.Data.SQLite;

    class sqlite_Main
    {
        public static void run(string[] args)
        {
            try
            {
                Console.WriteLine("******************* basic operation");
                BasicOperation.runTests();

                Console.WriteLine("\n******************* attaching multiple databases");
                AttachingMultipleDatabases.runTests();

                Console.WriteLine("\n******************* using ADO.NET");
                UsingAdoNet.runTests();

                Console.WriteLine("\n******************* using LINQ to SQL");
                UsingDLINQ.runTests();
            }
            catch (SQLiteException e)
            {
                Console.WriteLine("System.Data.SQLite.SQLiteException occurred: {0}", e);
            }
        }
    }
}
