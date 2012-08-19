using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Remoting.Messaging;

namespace NetRemotingObjects
{
    public class Hello : System.MarshalByRefObject
    {
        public Hello()
        {
            Console.WriteLine("Hello's constructor is called.");
        }
        ~Hello()
        {
            Console.WriteLine("Hello's destructor is called.");
        }

        public string greet(string name)
        {
            Console.WriteLine("Hello.greet() is called.");
            return "Hello, " + name;
        }

        public List<HelloMessage> greetAll(List<string> names)
        {
            Console.WriteLine("Hello.greetAll() is called.");

            List<HelloMessage> msgs = new List<HelloMessage>(names.Count);

            foreach (string name in names)
                msgs.Add(new HelloMessage("Hello, " + name));

            return msgs;
        }

        [OneWay]
        public void takeAWhile(int msec)
        {
            Console.WriteLine("Hello.takeAWhile() is started ...");
            System.Threading.Thread.Sleep(msec);
            Console.WriteLine("Hello.takeAWhile() is finished ...");
        }
    }
}
