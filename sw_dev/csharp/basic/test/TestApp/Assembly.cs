using System;
using System.Collections.Generic;
using System.Text;

namespace TestApp
{
    class Assembly
    {
        public static void run()
        {
            TestClassLibrary.Hello hello = new TestClassLibrary.Hello();
            hello.print();
            Console.Write(", ");
            TestClassLibrary.World world = new TestClassLibrary.World();
            world.print();
            Console.WriteLine(" !!!");

            Console.WriteLine("1 + 2 = " + TestClassLibrary.Adder.add(1, 2));
        }
    }
}
