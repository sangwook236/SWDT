using System;
using System.Collections.Generic;
using System.Text;

namespace TestApp
{
    class Program
    {
        static void Main(string[] args)
        {
			DebugAndTrace.run();

			String.run();
			//Collection.run();

            //IO.run();

            //Assembly.run();

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey(true);
        }
    }
}
