using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSharpBasic
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // data formatting
                Console.WriteLine("Data Formatting -------------------------------------------------------------");
                runDataFormatting();

                // date & time
                Console.WriteLine("\nDate & Time -----------------------------------------------------------------");
                runDateTime();

                // encoding
                Console.WriteLine("\nEncoding --------------------------------------------------------------------");
                Encoding.run();
            }
            catch (Exception e)
            {
                Console.WriteLine("System.Exception occurred: {0}", e);
            }

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey(true);
        }

        static void runDataFormatting()
        {
            int hex = 0x3E5;

            StringBuilder builder = new StringBuilder();

            builder.AppendFormat("{0:X}", (hex & 0xF));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:X2}", (hex & 0xF));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:X02}", (hex & 0xF));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:X02}", (hex & 0xFF));
            Console.WriteLine(builder.ToString());

            Console.WriteLine();

            //
            int dec = 654321;

            builder.Clear();
            builder.AppendFormat("{0:D01}", (dec % 10));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:D02}", (dec % 100));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:D03}", (dec % 1000));
            Console.WriteLine(builder.ToString());

            builder.Clear();
            builder.AppendFormat("{0:D04}", (dec % 10000));
            Console.WriteLine(builder.ToString());
        }

        static void runDateTime()
        {
            DateTime timestamp;
            //bool ret = DateTime.TryParse("2012-08-21 오후 4:45:37", out eventTimestamp);
            bool ret = DateTime.TryParse("2012-08-21 4:45:37 PM", out timestamp);

            Console.WriteLine("Timestamp: {0}", timestamp);
            Console.WriteLine("Now: {0}", DateTime.Now);
        }
    }
}
