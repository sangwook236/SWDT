#include <boost/log/support/date_time.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/utility/setup/from_stream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/common.hpp>
#include <boost/log/core.hpp>
#include <boost/smart_ptr.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace {
namespace local {

// REF [file] >> ${BOOST_HOME}/libs/log/examples/trivial/main.cpp
void trivial_log_example()
{
	// Trivial boost::log: all log records are written into a file.
	BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
	BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
	BOOST_LOG_TRIVIAL(info) << "An informational severity message";
	BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
	BOOST_LOG_TRIVIAL(error) << "An error severity message";
	BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";

	// Filtering can also be applied.
	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);

	// Now the first two lines will not pass the filter.
	BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
	BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
	BOOST_LOG_TRIVIAL(info) << "An informational severity message";
	BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
	BOOST_LOG_TRIVIAL(error) << "An error severity message";
	BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";
}

// Here we define our application severity levels.
enum severity_level
{
	normal,
	notification,
	warning,
	error,
	critical
};

// REF [file] >> ${BOOST_HOME}/libs/log/examples/basic_usage/main.cpp
void basic_usage()
{
	// The first thing we have to do to get using the library is to set up the boost::log sinks - i.e. where the logs will be written to.
	boost::log::add_console_log(std::clog, boost::log::keywords::format = "%TimeStamp%: %Message%");

	// One can also use lambda expressions to setup filters and formatters.
	boost::log::add_file_log
	(
		"./data/boost/sample.log",
		boost::log::keywords::filter = boost::log::expressions::attr<severity_level>("Severity") >= warning,
		boost::log::keywords::format = boost::log::expressions::stream
			<< boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d, %H:%M:%S.%f")
			<< " [" << boost::log::expressions::format_date_time<boost::log::attributes::timer::value_type>("Uptime", "%O:%M:%S")
			<< "] [" << boost::log::expressions::format_named_scope("Scope", boost::log::keywords::format = "%n (%f:%l)")
			<< "] <" << boost::log::expressions::attr<severity_level>("Severity")
			<< "> " << boost::log::expressions::message
		/*
		boost::log::keywords::format = boost::log::expressions::format("%1% [%2%] [%3%] <%4%> %5%")
			% boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d, %H:%M:%S.%f")
			% boost::log::expressions::format_date_time<boost::log::attributes::timer::value_type>("Uptime", "%O:%M:%S")
			% boost::log::expressions::format_named_scope("Scope", boost::log::keywords::format = "%n (%f:%l)")
			% boost::log::expressions::attr<severity_level>("Severity")
			% boost::log::expressions::message
		*/
	);

	// Also let's add some commonly used attributes, like timestamp and record counter.
	boost::log::add_common_attributes();
	boost::log::core::get()->add_thread_attribute("Scope", boost::log::attributes::named_scope());

	BOOST_LOG_FUNCTION();

	// Now our logs will be written both to the console and to the file.
	// Let's do a quick test and output something. We have to create a logger for this.
	boost::log::sources::logger logger;

	// And output...
	BOOST_LOG(logger) << "Hello, World!";

	// Now, let's try boost::log with severity.
	boost::log::sources::severity_logger<severity_level> slg;

	// Let's pretend we also want to profile our code, so add a special timer attribute.
	slg.add_attribute("Uptime", boost::log::attributes::timer());

	BOOST_LOG_SEV(slg, normal) << "A normal severity message, will not pass to the file";
	BOOST_LOG_SEV(slg, warning) << "A warning severity message, will pass to the file";
	BOOST_LOG_SEV(slg, error) << "An error severity message, will pass to the file";
}

// REF [file] >> ${BOOST_HOME}/libs/log/examples/rotating_file/main.cpp
void rotating_file()
{
	const size_t LOG_RECORDS_TO_WRITE = 10000;

	// Create a text file sink.
	typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_file_backend> file_sink;
	boost::shared_ptr<file_sink> sink(new file_sink(
		boost::log::keywords::file_name = "%Y%m%d_%H%M%S_%5N.log",  // File name pattern.
		boost::log::keywords::rotation_size = 16384  // Rotation size, in characters.
	));

	// Set up where the rotated files will be stored.
	sink->locked_backend()->set_file_collector(boost::log::sinks::file::make_collector(
		boost::log::keywords::target = "./data/boost/logs",  // Where to store rotated files.
		boost::log::keywords::max_size = 16 * 1024 * 1024,  // Maximum total size of the stored files, in bytes.
		boost::log::keywords::min_free_space = 100 * 1024 * 1024,  // Minimum free space on the drive, in bytes.
		boost::log::keywords::max_files = 512  // Maximum number of stored files.
	));

	// Upon restart, scan the target directory for files matching the file_name pattern.
	sink->locked_backend()->scan_for_files();

	sink->set_formatter
	(
		boost::log::expressions::format("%1%: [%2%] - %3%")
		% boost::log::expressions::attr<unsigned int>("RecordID")
		% boost::log::expressions::attr<boost::posix_time::ptime>("TimeStamp")
		% boost::log::expressions::smessage
	);

	// Add it to the core.
	boost::log::core::get()->add_sink(sink);

	// Add some attributes too.
	boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
	boost::log::core::get()->add_global_attribute("RecordID", boost::log::attributes::counter<unsigned int>());

	// Do some logging.
	boost::log::sources::logger logger;
	std::cout << "Start logging..." << std::endl;
	for (size_t i = 0; i < LOG_RECORDS_TO_WRITE; ++i)
	{
		BOOST_LOG(logger) << "Some log record";
	}
	std::cout << "End logging..." << std::endl;
}

//  Global logger declaration.
BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(test_logger, boost::log::sources::severity_logger<>)

// REF [file] >> ${BOOST_HOME}/libs/log/examples/settings_file/main.cpp
void settings_file()
{
	// Open the file.
	std::ifstream settings("./data/boost/boost_log_settings.txt");
	if (!settings.is_open())
	{
		std::cout << "Could not open ./data/boost/boost_log_settings.txt file" << std::endl;
		return;
	}

	// Read the settings and initialize logging library.
	boost::log::init_from_stream(settings);

	// Add some attributes.
	boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());

	// Try logging.
	{
		boost::log::sources::severity_logger<>& logger = test_logger::get();
		BOOST_LOG_SEV(logger, normal) << "This is a normal severity record";
		BOOST_LOG_SEV(logger, notification) << "This is a notification severity record";
		BOOST_LOG_SEV(logger, warning) << "This is a warning severity record";
		BOOST_LOG_SEV(logger, error) << "This is a error severity record";
		BOOST_LOG_SEV(logger, critical) << "This is a critical severity record";
	}

	// Now enable tagging and try again.
	BOOST_LOG_SCOPED_THREAD_TAG("Tag", "TAGGED");
	{
		boost::log::sources::severity_logger<>& logger = test_logger::get();
		BOOST_LOG_SEV(logger, normal) << "This is a normal severity record";
		BOOST_LOG_SEV(logger, notification) << "This is a notification severity record";
		BOOST_LOG_SEV(logger, warning) << "This is a warning severity record";
		BOOST_LOG_SEV(logger, error) << "This is a error severity record";
		BOOST_LOG_SEV(logger, critical) << "This is a critical severity record";
	}
}

}  // namespace local
}  // unnamed namespace

void log()
{
	//local::trivial_log_example();
	//local::basic_usage();
	local::rotating_file();
	//local::settings_file();
}
