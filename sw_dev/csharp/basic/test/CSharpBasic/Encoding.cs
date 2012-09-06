using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSharpBasic
{
    class Encoding
    {
        public static void run()
        {
            string str = "한글 변환 123 ABC abc";
            {
                //System.Text.Encoding encoding = new System.Text.ASCIIEncoding();
                System.Text.Encoding encoding = System.Text.Encoding.ASCII;
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("ASCII : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                //System.Text.Encoding encoding = new System.Text.UTF7Encoding();
                System.Text.Encoding encoding = System.Text.Encoding.UTF7;
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("UTF7 : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                //System.Text.Encoding encoding = new System.Text.UTF8Encoding();
                System.Text.Encoding encoding = System.Text.Encoding.UTF8;
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("UTF8 : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                //System.Text.Encoding encoding = new System.Text.UnicodeEncoding();
                System.Text.Encoding encoding = System.Text.Encoding.Unicode;
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("UTF16 : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                //System.Text.Encoding encoding = new System.Text.UTF32Encoding();
                System.Text.Encoding encoding = System.Text.Encoding.UTF32;
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("UTF32 : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                // [ref] http://msdn.microsoft.com/en-us/library/system.text.encoding.issinglebyte

                System.Text.Encoding encoding = System.Text.Encoding.GetEncoding(949);
                //System.Text.Encoding encoding = System.Text.Encoding.GetEncoding("ks_c_5601-1987");
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("ks_c_5601-1987 : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                // [ref] http://msdn.microsoft.com/en-us/library/system.text.encoding.issinglebyte

                System.Text.Encoding encoding = System.Text.Encoding.GetEncoding(51949);
                //System.Text.Encoding encoding = System.Text.Encoding.GetEncoding("euc-kr");
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("euc-kr : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                // [ref] http://msdn.microsoft.com/en-us/library/system.text.encoding.issinglebyte

                System.Text.Encoding encoding = System.Text.Encoding.GetEncoding(50225);
                //System.Text.Encoding encoding = System.Text.Encoding.GetEncoding("iso-2022-kr");
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("iso-2022-kr : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                // [ref] http://msdn.microsoft.com/en-us/library/system.text.encoding.issinglebyte

                System.Text.Encoding encoding = System.Text.Encoding.GetEncoding(20833);
                //System.Text.Encoding encoding = System.Text.Encoding.GetEncoding("x-EBCDIC-KoreanExtended");
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("x-EBCDIC-KoreanExtended : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }

            {
                // [ref] http://msdn.microsoft.com/en-us/library/system.text.encoding.issinglebyte

                System.Text.Encoding encoding = System.Text.Encoding.GetEncoding(10003);
                //System.Text.Encoding encoding = System.Text.Encoding.GetEncoding("x-mac-korean");
                byte[] bytes = encoding.GetBytes(str);
                Console.Write("x-mac-korean : Len = {0} : ", bytes.Length);
                foreach (byte item in bytes)
                    Console.Write("{0:X02} ", item);
                Console.WriteLine();
            }
        }
    }
}
