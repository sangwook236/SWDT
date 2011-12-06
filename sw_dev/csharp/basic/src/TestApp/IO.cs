using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace TestApp
{
    class IO
    {
        public static void run()
        {
            runFileInfo();
            runFile();
            runDirectory();
            runPath();

            runStreamReaderAndStreamWriter();
            runFileStream();
        }

        static void runFileInfo()
        {
            string path = Path.GetTempFileName();
            FileInfo fi = new FileInfo(path);

            if (!fi.Exists)
            {
                // create a file to write to.
                using (StreamWriter writer = fi.CreateText())
                {
                    writer.WriteLine("Hello");
                    writer.WriteLine("And");
                    writer.WriteLine("Welcome");
                }
            }

            // open the file to read from.
            using (StreamReader reader = fi.OpenText())
            {
                string s = "";
                while ((s = reader.ReadLine()) != null)
                {
                    Console.WriteLine(s);
                }
            }
        }

        static void runFile()
        {
            //string path = @"..\data\file_test.txt";
            string path = Path.GetTempFileName();

            if (!File.Exists(path))
            {
                // create a file to write to.
                using (StreamWriter sw = File.CreateText(path))
                {
                    sw.WriteLine("Hello");
                    sw.WriteLine("And");
                    sw.WriteLine("Welcome");
                }
            }

            // open the file to read from.
            using (StreamReader sr = File.OpenText(path))
            {
                string s = "";
                while ((s = sr.ReadLine()) != null)
                {
                    Console.WriteLine(s);
                }
            }
        }

        static void runDirectory()
        {
            string pwd = System.IO.Directory.GetCurrentDirectory();
            Console.WriteLine(pwd);
        }

        static void runPath()
        {
            string path1 = @"..\data\MyTest.txt";
            string path2 = @"..\data\MyTest";
            string path3 = @"temp";

            if (Path.HasExtension(path1))
            {
                Console.WriteLine("{0} has an extension.", path1);
            }

            if (!Path.HasExtension(path2))
            {
                Console.WriteLine("{0} has no extension.", path2);
            }

            if (!Path.IsPathRooted(path3))
            {
                Console.WriteLine("The string {0} contains no root information.", path3);
            }

            Console.WriteLine("The full path of {0} is {1}.", path3, Path.GetFullPath(path3));
            Console.WriteLine("{0} is the location for temporary files.", Path.GetTempPath());
            Console.WriteLine("{0} is a file available for use.", Path.GetTempFileName());
        }

        static void runStreamReaderAndStreamWriter()
        {
            //TextWriter writer = new StreamWriter("..\\data\\file_io_test.txt");  // Oops !!! error
            TextWriter writer = new StreamWriter("..\\data\\file_io_test.txt", false);
            writer.WriteLine("{0}\t{1}\t{2}\t{3}", 100, -200, 300, -400);
            writer.Close();
            writer = null;

            TextReader reader = new StreamReader("..\\data\\file_io_test.txt");
            Console.WriteLine("read all the characters =");
            int ch = -1;
            while ((ch = reader.Read()) != -1)
                Console.Write(Convert.ToChar(ch).ToString());
            Console.WriteLine();
            reader.Close();

            reader = new StreamReader("..\\data\\file_io_test.txt");
            Console.WriteLine("read all the lines =");
            //while (!reader.EndOfStream)
            //while (reader.Peek() >= 0)
            //    Console.WriteLine(reader.ReadLine());
            string s = "";
            while ((s = reader.ReadLine()) != null)
                Console.WriteLine(s);
            reader.Close();
            reader = null;
        }

        private static void addText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

        static void runFileStream()
        {
            string path = @"..\\data\\filestream_test.txt";

            // delete the file if it exists.
            if (File.Exists(path))
            {
                File.Delete(path);
            }

            // create the file.
            using (FileStream stream = File.Create(path))
            {
                addText(stream, "This is some text");
                addText(stream, "This is some more text,");
                addText(stream, "\r\nand this is on a new line");
                addText(stream, "\r\n\r\nThe following is a subset of characters:\r\n");

                for (int i = 1; i < 120; ++i)
                {
                    addText(stream, Convert.ToChar(i).ToString());

                    // split the output at every 10th character.
                    if (Math.IEEERemainder(Convert.ToDouble(i), 10) == 0)
                    {
                        addText(stream, "\r\n");
                    }
                }
            }

            // open the stream and read it back.
            using (FileStream stream = File.OpenRead(path))
            {
                byte[] b = new byte[1024];
                UTF8Encoding encoding = new UTF8Encoding(true);
                while (stream.Read(b, 0, b.Length) > 0)
                {
                    Console.WriteLine(encoding.GetString(b));
                }
            }
        }
    }
}
