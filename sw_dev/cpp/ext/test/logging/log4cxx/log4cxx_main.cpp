#include <limits>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/test/test_tools.hpp>
#include <log4cxx/logger.h>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/xml/domconfigurator.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/rollingfileappender.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/helpers/pool.h>


#if defined(max)
#	undef max
#endif


namespace {
namespace local {

log4cxx::LoggerPtr rootLogger(log4cxx::Logger::getRootLogger());
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger(L"swlLogger.logger"));
log4cxx::LoggerPtr tracer(log4cxx::Logger::getLogger(L"swlLogger.tracer"));
boost::mutex mutex;

log4cxx::AppenderPtr create_appender(const bool to_file = true)
{
	log4cxx::PatternLayoutPtr layout(new log4cxx::PatternLayout());
	//layout->setConversionPattern("[%5p][%F:%L][%d{ABSOLUTE}] [DIA] %m%n");
	layout->setConversionPattern("[%5p][%F:%L][%d{yyyy-MM-ddTHH:mm:ss}] [DIA] %m%n");

	log4cxx::AppenderPtr appender = nullptr;
	if (to_file)
	{
		log4cxx::RollingFileAppenderPtr fileAppender(new log4cxx::RollingFileAppender());
		fileAppender->setName("fileAppender");
		fileAppender->setFile("./log/dia.log");
		fileAppender->setAppend(true);
		fileAppender->setMaxFileSize("10MB");
		fileAppender->setMaxBackupIndex(10);
		fileAppender->setLayout(layout);

		appender = fileAppender;
	}
	else
	{
		log4cxx::ConsoleAppenderPtr consoleAppender(new log4cxx::ConsoleAppender());
		consoleAppender->setName("consoleAppender");
		consoleAppender->setTarget("System.out");
		consoleAppender->setImmediateFlush(true);
		consoleAppender->setLayout(layout);

		appender = consoleAppender;
	}

	log4cxx::helpers::Pool p;
	appender->activateOptions(p);

	return appender;
}

inline boost::xtime delay(int secs, int msecs = 0, int nsecs = 0)
{
    const int MILLISECONDS_PER_SECOND = 1000;
    const int NANOSECONDS_PER_SECOND = 1000000000;
    const int NANOSECONDS_PER_MILLISECOND = 1000000;

    boost::xtime xt;
    //BOOST_CHECK_EQUAL(static_cast<int>(boost::xtime_get(&xt, boost::TIME_UTC_)), static_cast<int>(boost::TIME_UTC_));
	boost::xtime_get(&xt, boost::TIME_UTC_);

    nsecs += xt.nsec;
    msecs += nsecs / NANOSECONDS_PER_MILLISECOND;
    secs += msecs / MILLISECONDS_PER_SECOND;
    nsecs += (msecs % MILLISECONDS_PER_SECOND) * NANOSECONDS_PER_MILLISECOND;
    xt.nsec = nsecs % NANOSECONDS_PER_SECOND;
    xt.sec += secs + (nsecs / NANOSECONDS_PER_SECOND);

    return xt;
}

void thread_func()
{
	//boost::lock_guard<boost::mutex> guard(mutex);

	LOG4CXX_WARN(logger, L"Low fuel level.");
	LOG4CXX_ERROR(tracer, L"Located nearest gas station.");

	//boost::thread::sleep(delay(0, 500, 0));
	boost::thread::yield();

	LOG4CXX_DEBUG(logger, L"Starting search for nearest gas station.");
	LOG4CXX_DEBUG(tracer, L"Exiting gas station search.");
}

}  // namespace local
}  // unnamed namespace

namespace my_log4cxx {

}  // namespace my_log4cxx

int log4cxx_main(int argc, char *argv[])
{
	try
	{
		std::wcout << L"Level ALL: " << log4cxx::Level::getAll()->toInt() << std::endl;
		std::wcout << L"Level TRACE: " << log4cxx::Level::getTrace()->toInt() << std::endl;
		std::wcout << L"Level DEBUG: " << log4cxx::Level::getDebug()->toInt() << std::endl;
		std::wcout << L"Level INFO: " << log4cxx::Level::getInfo()->toInt() << std::endl;
		std::wcout << L"Level WARN: " << log4cxx::Level::getWarn()->toInt() << std::endl;
		std::wcout << L"Level ERROR: " << log4cxx::Level::getError()->toInt() << std::endl;
		std::wcout << L"Level FATAL: " << log4cxx::Level::getFatal()->toInt() << std::endl;
		std::wcout << L"Level OFF: " << log4cxx::Level::getOff()->toInt() << std::endl;
		std::wcout << std::endl;

		//
		std::wcout << L"***** Start configuring." << std::endl;

		const int config = 1;
		switch (config)
		{
		case 1:
			log4cxx::PropertyConfigurator::configure(L"data/logging/log4cxx/swl_logger_conf.properties");
			break;
		case 2:
			log4cxx::xml::DOMConfigurator::configure(L"data/logging/log4cxx/swl_logger_conf.xml");  // Run-time error.
			break;
		case 0:
		default:
			//log4cxx::BasicConfigurator::configure();
			log4cxx::BasicConfigurator::configure(local::create_appender(true));
			log4cxx::BasicConfigurator::configure(local::create_appender(false));
			break;
		}

		// If this variable 'additivity' is set to false then the appenders found in the ancestors of this logger are not used.
		// However, the children of this logger will inherit its appenders, unless the children have their additivity flag set to false too. 
		//logger->setAdditivity(false);

		//logger->setLevel(log4cxx::Level::getInfo());
		std::wcout << L"***** End configuring." << std::endl;

		//
		std::wcout << L"***** Start processinng." << std::endl;

		const int num_threads = 4;

		boost::thread_group threads;
		for (int i = 0; i < num_threads; ++i)
			threads.create_thread(&local::thread_func);

		threads.join_all();

		std::wcout << L"***** End processinng." << std::endl;
	}
	catch (const log4cxx::helpers::Exception &ex)
	{
		std::cerr << "log4cxx::helpers::Exception caught: " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
