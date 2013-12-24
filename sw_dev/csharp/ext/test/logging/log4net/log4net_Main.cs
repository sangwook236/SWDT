using System;
using System.Collections.Generic;
using System.Text;

namespace logging
{
    using log4net;
    using log4net.Config;

    class log4net_Main
    {
        public static void run(string[] args) 
        {
		    const int config = 1;
            switch (config)
            {
                case 1:
                    XmlConfigurator.Configure(new System.IO.FileInfo("..\\data\\logging\\log4net\\swl_logger_conf.xml"));
                    break;
                default:
                    BasicConfigurator.Configure();
                    break;
            }

            logger_.Info("Entering application.");
            Logger.getDefaultLogger().Warn("Entering application.");
            Bar bar = new Bar();
            bar.run();
            logger_.Info("Exiting application.");
            Logger.getDefaultLogger().Warn("Exiting application.");
        }

        //private static readonly ILog logger_ = LogManager.GetLogger(typeof(Program));
        //private static readonly ILog logger_ = LogManager.GetLogger("swlLogger.tracer");
        private static readonly ILog logger_ = LogManager.GetLogger("swlLogger.logger");
    }

    class Bar
    {
        public void run()
        {
            logger_.Debug("Did it again!");
            logger_.Warn("Did it again!");
            Logger.getDefaultLogger().Debug("Did it again!");
            Logger.getDefaultLogger().Warn("Did it again!");
        }

        //private static readonly ILog logger_ = LogManager.GetLogger(typeof(Bar));
        private static readonly ILog logger_ = LogManager.GetLogger("swlLogger.tracer");
    }

    class Logger
    {
        static Logger()
        {
            try
            {
                log4net.Config.XmlConfigurator.Configure(new System.IO.FileInfo("..\\data\\logging\\log4net\\swl_logger_conf.xml"));
            }
            catch (Exception e)
            {
                log4net.Config.BasicConfigurator.Configure();

                getDefaultLogger().Fatal("logger configuration error: " + e.Message);
            }
        }

        public static ILog getDefaultLogger()
        {
            string appPath = Environment.GetCommandLineArgs()[0];
            int idx = appPath.LastIndexOf('\\');
            string appFileName = appPath.Substring(idx + 1);

            idx = appFileName.LastIndexOf('.');
            string appName = appFileName.Substring(0, idx);
            string appFileExt = appFileName.Substring(idx + 1);

            idx = appName.LastIndexOf('.');
            if (appName.Substring(idx + 1).ToLower().CompareTo("vshost") == 0)
                appName = appName.Substring(0, idx);

            return string.IsNullOrEmpty(appName) ? log4net.LogManager.GetLogger(typeof(Logger)) : log4net.LogManager.GetLogger(appName + ".Logger");
        }
    }
}
