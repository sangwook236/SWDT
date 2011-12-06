#include <log4cxx/Logger.h>
#include <log4cxx/BasicConfigurator.h>
#include <log4cxx/PropertyConfigurator.h>
#include <log4cxx/xml/DOMConfigurator.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/test/test_tools.hpp>
#include <iostream>
#include <limits>


#if defined(_MSC_VER) && defined(_DEBUG)
#define VC_EXTRALEAN  //  Exclude rarely-used stuff from Windows headers
//#include <afx.h>
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


#if defined(max)
#	undef max
#endif


log4cxx::LoggerPtr rootLogger(log4cxx::Logger::getRootLogger());
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger(L"swlLogger.logger"));
log4cxx::LoggerPtr tracer(log4cxx::Logger::getLogger(L"swlLogger.tracer"));
boost::mutex mutex;


inline boost::xtime delay(int secs, int msecs = 0, int nsecs = 0)
{
    const int MILLISECONDS_PER_SECOND = 1000;
    const int NANOSECONDS_PER_SECOND = 1000000000;
    const int NANOSECONDS_PER_MILLISECOND = 1000000;

    boost::xtime xt;
    //BOOST_CHECK_EQUAL(static_cast<int>(boost::xtime_get(&xt, boost::TIME_UTC)), static_cast<int>(boost::TIME_UTC));
	boost::xtime_get(&xt, boost::TIME_UTC);

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

int main()
{
	try
	{
		//
		std::wcout << L"level ALL: " << log4cxx::Level::getAll()->toInt() << std::endl;
		std::wcout << L"level TRACE: " << log4cxx::Level::getTrace()->toInt() << std::endl;
		std::wcout << L"level DEBUG: " << log4cxx::Level::getDebug()->toInt() << std::endl;
		std::wcout << L"level INFO: " << log4cxx::Level::getInfo()->toInt() << std::endl;
		std::wcout << L"level WARN: " << log4cxx::Level::getWarn()->toInt() << std::endl;
		std::wcout << L"level ERROR: " << log4cxx::Level::getError()->toInt() << std::endl;
		std::wcout << L"level FATAL: " << log4cxx::Level::getFatal()->toInt() << std::endl;
		std::wcout << L"level OFF: " << log4cxx::Level::getOff()->toInt() << std::endl;
		std::wcout << std::endl;

		//
		std::wcout << L"************************************** start configuring" << std::endl;

		const int config = 2;
		switch (config)
		{
		case 1:
			log4cxx::PropertyConfigurator::configure(L"log4cxx_data\\swl_logger_conf.properties");
			break;
		case 2:
			log4cxx::xml::DOMConfigurator::configure(L"log4cxx_data\\swl_logger_conf.xml");
			break;
		case 0:
		default:
			log4cxx::BasicConfigurator::configure();
			break;
		}

		//logger->setLevel(log4cxx::Level::getInfo());
		std::wcout << L"************************************** end configuring" << std::endl;

		//
		std::wcout << L"************************************** start procesinng" << std::endl;

		const int num_threads = 4;

		boost::thread_group threads;
		for (int i=0; i < num_threads; ++i)
			threads.create_thread(&thread_func);

		threads.join_all();

		std::wcout << L"************************************** end procesinng" << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cerr << "exception caught: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "unknown exception caught ..." << std::endl;
	}

	std::cout << "press any key to terminate ..." << std::endl;
	std::wcin.get();
	return 0;
}
