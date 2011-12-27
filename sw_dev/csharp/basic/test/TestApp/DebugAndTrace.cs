using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace TestApp
{
	class DebugAndTrace
	{
		public static void run()
		{
			Console.WriteLine(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Debug");
			runDebug();
			Console.WriteLine("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Trace");
			runTrace();
		}

		static void runDebug()
		{
			Debug.Listeners.Add(new TextWriterTraceListener(Console.Out));
			Debug.AutoFlush = true;
			Debug.Indent();
			Debug.WriteLine("Debug: Entering runDebug()");
			Console.WriteLine("Debug: Hello !!! Debug.");
			Debug.WriteLine("Debug: Exiting runDebug()");
			Debug.Unindent();
		}

		static void runTrace()
		{
			Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));
			Trace.AutoFlush = true;
			Trace.Indent();
			Trace.WriteLine("Trace: Entering runTrace()");
			Console.WriteLine("Trace: Hello !!! Trace.");
			Trace.WriteLine("Trace: Exiting runTrace()");
			Trace.Unindent();
		}
	}
}
