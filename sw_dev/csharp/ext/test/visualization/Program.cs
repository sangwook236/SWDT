using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace visualization
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("OxyPlot library -----------------------------------------------------");
                oxyplot.OxyPlot_Main.run(args);
 
                Console.WriteLine("Dynamic Data Display (D3) library -----------------------------------");
                d3.D3_Main.run(args);
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
