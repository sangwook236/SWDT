using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;

namespace NetRemotingClient
{
    class Program
    {
        private delegate string GreetDelegate(string name);
        private delegate List<NetRemotingObjects.HelloMessage> GreetAllDelegate(List<string> names);

        static void Main(string[] args)
        {
            bool ensureSecurity = false;
            ChannelServices.RegisterChannel(new TcpClientChannel(), ensureSecurity);

            NetRemotingObjects.Hello obj = (NetRemotingObjects.Hello)Activator.GetObject(typeof(NetRemotingObjects.Hello), "tcp://localhost:8086/Hi");
            if (null != obj)
            {
                string name = "Sang-Wook Lee";
                List<string> names = new List<string>();

                names.Add("Hye-Joong Kim");
                names.Add("Chae-Wan Lee");
                names.Add("Ji-Hyoung Lee");
#if false
                runSynchronousCall(obj, name, names);
#else
                runAsynchronousCall(obj, name, names);
#endif

                obj.takeAWhile(3000);
            }
            else
            {
                Console.WriteLine("a .NET Remoting server not found");
            }

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey(true);
        }

        static void runSynchronousCall(NetRemotingObjects.Hello obj, string name, List<string> names)
        {
            Console.WriteLine(obj.greet(name));

            List<NetRemotingObjects.HelloMessage> msgs = obj.greetAll(names);
            foreach (NetRemotingObjects.HelloMessage msg in msgs)
                Console.WriteLine(msg.Message);
        }

        static void runAsynchronousCall(NetRemotingObjects.Hello obj, string name, List<string> names)
        {
            // calling synchronous methods asynchronously.
            //  [ref] http://msdn.microsoft.com/en-us/library/2e08f6yc.aspx

            GreetDelegate greetDelegate = new GreetDelegate(obj.greet);
            IAsyncResult greetAsyncResult = greetDelegate.BeginInvoke(name, null, null);
            GreetAllDelegate greetAllDelegate = new GreetAllDelegate(obj.greetAll);
            IAsyncResult greetAllAsyncResult = greetAllDelegate.BeginInvoke(names, null, null);

            // do something and then wait

            greetAsyncResult.AsyncWaitHandle.WaitOne();
            greetAllAsyncResult.AsyncWaitHandle.WaitOne();

            string greeting = null;
            if (greetAsyncResult.IsCompleted)
                greeting = greetDelegate.EndInvoke(greetAsyncResult);
            List<NetRemotingObjects.HelloMessage> greetingMsgs = null;
            if (greetAllAsyncResult.IsCompleted)
                greetingMsgs = greetAllDelegate.EndInvoke(greetAllAsyncResult);

            if (null != greeting)
                Console.WriteLine(greeting);
            else
                Console.WriteLine("NetRemotingObjects.Hello.greet() call error");
            if (null != greetingMsgs)
                foreach (NetRemotingObjects.HelloMessage msg in greetingMsgs)
                    Console.WriteLine(msg.Message);
            else
                Console.WriteLine("NetRemotingObjects.Hello.greetAll() call error");
        }
    }
}
