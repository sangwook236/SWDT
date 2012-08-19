using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;

namespace NetRemotingServer
{
    class Program
    {
        static void Main(string[] args)
        {
            bool ensureSecurity = false;
            TcpServerChannel channel = new TcpServerChannel(8086);
            ChannelServices.RegisterChannel(channel, ensureSecurity);
#if true
            RemotingConfiguration.RegisterWellKnownServiceType(typeof(NetRemotingObjects.Hello), "Hi", WellKnownObjectMode.SingleCall);
#else
            RemotingConfiguration.RegisterWellKnownServiceType(typeof(NetRemotingObjects.Hello), "Hi", WellKnownObjectMode.Singleton);
#endif

            Console.WriteLine("a .NET Remoting server is running ...");

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey(true);
        }
    }
}
