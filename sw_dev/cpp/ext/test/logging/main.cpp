#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#if defined(_WIN64) || defined(_WIN32)
#include <windows.h>
#include <winevt.h>
#else
#include <syslog.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


namespace {
namespace local {

#if defined(_WIN64) || defined(_WIN32)
#pragma comment(lib, "wevtapi.lib")

// REF [site] >>
//	Windows Event Logging (recommended): https://msdn.microsoft.com/en-us/library/windows/desktop/aa385780(v=vs.85).aspx
//	Event Logging (deprecated): https://msdn.microsoft.com/en-us/library/windows/desktop/aa363652(v=vs.85).aspx

// REF [site] >> https://msdn.microsoft.com/en-us/library/windows/desktop/dd996928(v=vs.85).aspx
void event_log_example()
{
	throw std::runtime_error("Not yet implemented");
}
#else
// Config file: /etc/rsyslog.conf or /etc/syslog.conf.
// Log file: /var/log/syslog.
//	tail -f /var/log/syslog
//	tail -f /var/log/syslog | grep process_name
//	less /var/log/syslog

// REF [site] >>
//	https://www.gnu.org/software/libc/manual/html_node/Syslog.html
//	https://www.gnu.org/software/libc/manual/html_node/Submitting-Syslog-Messages.html
void syslog_example()
{
	//setlogmask(LOG_MASK(LOG_EMERG) | LOG_MASK(LOG_ERROR));
	//setlogmask(~(LOG_MASK(LOG_INFO)));
	setlogmask(LOG_UPTO(LOG_NOTICE));

	openlog("exampleprog", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL1);

	syslog(LOG_NOTICE, "Program started by User %d", getuid());
	syslog(LOG_MAKEPRI(LOG_USER, LOG_INFO), "A tree falls in a forest");

	closelog();
}
#endif

}  // namespace local
}  // unnamed namespace

int main(int argc, char *argv[])
{
	int log4cxx_main(int argc, char *argv[]);
	int glog_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
#if defined(_WIN64) || defined(_WIN32)
		std::cout << "Windows Event Log ---------------------------------------------------" << std::endl;
		//local::event_log_example();  // Not yet implemented.
#else
		std::cout << "syslog --------------------------------------------------------------" << std::endl;
		local::syslog_example();
#endif

		std::cout << "\nBoost.Log libary ----------------------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nlog4cxx libary ------------------------------------------------------" << std::endl;
		//retval = log4cxx_main(argc, argv);

		std::cout << "\nglog library --------------------------------------------------------" << std::endl;
		retval = glog_main(argc, argv);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
