using System;
using System.Collections.Generic;
using System.Text;

namespace log4net_test
{
    using log4net;
    using log4net.Config;

    class Program
    {
        static void Main(string[] args) 
        {
		    const int config = 1;
            switch (config)
            {
                case 1:
                    XmlConfigurator.Configure(new System.IO.FileInfo("..\\log4net_data\\swl_logger_conf.xml"));
                    break;
                default:
                    BasicConfigurator.Configure();
                    break;
            }

            logger_.Info("Entering application.");
            Bar bar = new Bar();
            bar.run();
            logger_.Info("Exiting application.");
        }

        //private static readonly ILog logger_ = LogManager.GetLogger(typeof(Program));
        private static readonly ILog logger_ = LogManager.GetLogger("swlLogger.logger");
    }

    class Bar
    {
        public void run()
        {
            logger_.Debug("Did it again!");
        }

        //private static readonly ILog logger_ = LogManager.GetLogger(typeof(Bar));
        private static readonly ILog logger_ = LogManager.GetLogger("swlLogger.tracer");
    }
}
