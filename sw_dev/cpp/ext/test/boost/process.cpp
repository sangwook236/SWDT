#include <boost/process.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <thread>
#include <chrono>
#include <iostream>


namespace {
namespace local {

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void system_example()
{
	//const int exitCode = std::system("g++ data/boost/main.cpp");
	//const int exitCode = boost::process::system("g++ data/boost/main.cpp");
	//std::error_code errorCode;
	//const int exitCode = boost::process::system("g++ data/boost/main.cpp", errorCode);
	//const int exitCode = boost::process::system("g++ data/boost/main.cpp", boost::process::std_out > stdout, boost::process::std_err > stderr, boost::process::std_in < stdin);
	//const int exitCode = boost::process::system("g++ data/boost/main.cpp", boost::process::std_out > boost::process::null);
	//const int exitCode = boost::process::system("g++ data/boost/main.cpp", boost::process::std_out > "gcc_out.log");

#if defined(_WIN64) || defined(_WIN32)
	const std::string exec_path("D:/MyProgramFiles/cygwin64/bin/g++.exe");
#else
	const std::string exec_path("/usr/bin/g++");
#endif
	//const int exitCode = boost::process::system(exec_path, "data/boost/main.cpp");

	//const boost::filesystem::path p(exec_path);
	const boost::filesystem::path p(boost::process::search_path("g++"));
	//std::cout << "Path = " << p.string() << std::endl;
	const int exitCode = boost::process::system(p, "data/boost/main.cpp");

	std::cout << "Exit code = " << exitCode << std::endl;
}

void async_system_example()
{
	boost::asio::io_service ios;
	std::function<void(const boost::system::error_code &, int)> exit_handler([](const boost::system::error_code &ec, int) {
		// Do something.
	});

	boost::process::async_system(ios, exit_handler, "g++ data/boost/main.cpp");

	ios.run();

	// FIXME [check] >> This implementation is temporary. Is it correct?
	//boost::asio::async_result<std::function<void(const boost::system::error_code &, int)> > ec(exit_handler);
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void spawn_example()
{
	// The spawn() function launches a process and immediately detached it, so no handle will be returned and the process will be ignored.
#if defined(_WIN64) || defined(_WIN32)
	boost::process::spawn(boost::filesystem::path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"), "www.boost.org");
#else
	boost::process::spawn(boost::process::search_path("chrome"), "www.boost.org");
#endif
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void sync_child_example1()
{
#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a");
	boost::process::child c(boost::process::search_path("ls"), "-la");
#else
	boost::process::child c(boost::process::search_path("ls"), "-la");
#endif

	while (c.running())
	{
		// Do something.
	}

	c.wait();
	//if (!c.wait_for(std::chrono::seconds(3)))
	//{
	//	std::cerr << "Timeout." << std::endl;
	//}
	//if (c.wait_until(std::chrono::system_clock::now() + std::chrono::seconds(3)))
	//{
	//	std::cerr << "Timeout." << std::endl;
	//}

	std::cout << "Exit code = " << c.exit_code() << std::endl;
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
// REF [file] >> ${BOOST_HOME}/libs/process/example/sync_io.cpp
void sync_child_example2()
{
	boost::process::ipstream is;  // Pipe stream.

#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a", boost::process::std_out > is);
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > is);
#else
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > is);
#endif

	std::vector<std::string> data;
	std::string line;
	while (c.running())
	{
		std::getline(is, line);
		if (!line.empty()) data.push_back(line);
	}

	c.wait();

	std::cout << "Exit code = " << c.exit_code() << std::endl;
	std::cout << "Count = " << data.size() << std::endl;
	if (!data.empty())
	{
		std::copy(data.begin(), data.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
		std::cout << std::endl;
	}
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void sync_child_example3()
{
	boost::process::pipe p;  // Pipe.
	boost::process::ipstream is;  // Pipe stream.

#if defined(_WIN64) || defined(_WIN32)
	const std::string filepath("D:/work/SWL_github/cpp/lib64/swl_base.lib");
#else
	const std::string filepath("/home/sangwook/work/SWL_github/cpp/lib64/libswl_base.so");
#endif

	boost::process::child nm(boost::process::search_path("nm"), filepath, boost::process::std_out > p);
	boost::process::child filt(boost::process::search_path("c++filt"), boost::process::std_in < p, boost::process::std_out > is);

	std::vector<std::string> data;
	std::string line;
	// When nm finished the pipe closes and c++filt exits.
	while (filt.running())
	{
		std::getline(is, line);
		if (!line.empty()) data.push_back(line);
	}

	nm.wait();
	filt.wait();

	std::cout << "Exit code = " << filt.exit_code() << std::endl;
	std::cout << "Count = " << data.size() << std::endl;
	if (!data.empty())
	{
		std::copy(data.begin(), data.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
		std::cout << std::endl;
	}
}

// REF [file] >> ${BOOST_HOME}/libs/process/example/async_io.cpp
void async_child_example1()
{
	boost::asio::io_service ios;
	boost::asio::streambuf buf;

#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a", boost::process::std_out > buf, ios);
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > buf, ios);
#else
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > buf, ios);
#endif

	// FIXME [check] >> An error occurred in debug mode.
	ios.run();
	// TODO [check] >> Exit code is changed.
	//c.wait();  // Passing an instance of boost::asio::io_service to the launching function automatically cause it to wait asynchronously for the exit, so no call of wait is needed.

	std::cout << "Exit code = " << c.exit_code() << std::endl;
	if (buf.size())
	{
		std::istream is(&buf);
		std::string line;
		while (is)
		{
			std::getline(is, line);
			std::cout << line << std::endl;
		}
	}
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void async_child_example2()
{
	boost::asio::io_service ios;
	std::vector<char> buf(128);
	boost::process::async_pipe ap(ios);

#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a", boost::process::std_out > ap);
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > ap);
#else
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > ap);
#endif

	boost::asio::async_read(ap, boost::asio::buffer(buf), [](const boost::system::error_code &ec, size_t size) {});

	ios.run();
	//c.wait();  // Passing an instance of boost::asio::io_service to the launching function automatically cause it to wait asynchronously for the exit, so no call of wait is needed.

	std::cout << "Exit code = " << c.exit_code() << std::endl;
	std::cout << "Count = " << buf.size() << std::endl;
	if (!buf.empty())
	{
		std::copy(buf.begin(), buf.end(), std::ostream_iterator<char>(std::cout, ""));
		std::cout << std::endl;
	}

#if defined(_DEBUG)
	// For debug mode.
	ap.close();
#endif
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void async_child_example3()
{
	boost::asio::io_service ios;
	std::vector<char> buf(128);

#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a", boost::process::std_out > boost::asio::buffer(buf), ios);
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > boost::asio::buffer(buf), ios);
#else
	boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_out > boost::asio::buffer(buf), ios);
#endif

	// FIXME [check] >> An error occurred in debug mode.
	ios.run();
	// TODO [check] >> Exit code is changed.
	//c.wait();  // Passing an instance of boost::asio::io_service to the launching function automatically cause it to wait asynchronously for the exit, so no call of wait is needed.

	std::cout << "Exit code = " << c.exit_code() << std::endl;
	std::cout << "Count = " << buf.size() << std::endl;
	if (!buf.empty())
	{
		std::copy(buf.begin(), buf.end(), std::ostream_iterator<char>(std::cout, ""));
		std::cout << std::endl;
	}
}

// REF [site] >> http://www.boost.org/doc/libs/1_65_1/doc/html/boost_process/tutorial.html
void async_child_example4()
{
	boost::asio::io_service ios;
	std::future<std::string> data;

#if defined(_WIN64) || defined(_WIN32)
	//boost::process::child c(boost::process::search_path("dir"), "/a", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
	//boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
	boost::process::child c(boost::process::search_path("ls"), "-la", "*.exe", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
	//boost::process::child c("ls -la *.exe", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
#else
	//boost::process::child c(boost::process::search_path("ls"), "-la", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
	boost::process::child c(boost::process::search_path("ls"), "-la", "*.exe", boost::process::std_in.close(), boost::process::std_out > data, boost::process::std_err > boost::process::null, ios);
#endif

	// FIXME [check] >> An error occurred in debug mode.
	ios.run();

	data.wait();  // Wait for result.
	// TODO [check] >> Exit code is changed.
	c.wait();  // Passing an instance of boost::asio::io_service to the launching function automatically cause it to wait asynchronously for the exit, so no call of wait is needed.
	//if (!c.wait_for(std::chrono::nanoseconds(1)))
	//	std::cout << "Timeout." << std::endl;

	std::cout << "Exit code = " << c.exit_code() << std::endl;
	const std::string str(data.get());  // Can access only once.
	if (!str.empty())
		std::cout << str << std::endl;
}

void process_group1()
{
	boost::process::group g;
	boost::process::child c("make", g);
	if (!g.wait_for(std::chrono::seconds(5)))
		g.terminate();

	c.wait();  // Avoid a zombie process and get the exit code.
}

void process_group2()
{
	boost::process::group g;
	boost::process::spawn("task1", g);
	boost::process::spawn("task2", g);

	// Do something.

	g.wait();
	//if (!g.wait_for(std::chrono::seconds(5)))
	//	g.terminate();
}

void environment()
{
	auto env = boost::this_process::environment();

	// Add a variable to the current environment.
	env["VALUE_1"] = "foo";

	// Copy it into an environment separate to the one of this process.
	boost::process::environment new_env = env;

	// Append a variable to the current environment.
	new_env["VALUE_2"] = { "bar1", "bar2" };

	// Launch a process with 'new_env'.
	boost::process::system("stuff", new_env);

	//
	boost::process::system("stuff", boost::process::env["VALUE_1"] = "foo", boost::process::env["VALUE_2"] += { "bar1", "bar2" });
}

}  // namespace local
}  // unnamed namespace

void process()
{
	//local::system_example();
	//local::async_system_example();  // Not yet finished.

	//local::spawn_example();

	//local::sync_child_example1();
	//local::sync_child_example2();
	//local::sync_child_example3();

	//local::async_child_example1();
	//local::async_child_example2();
	//local::async_child_example3();
	local::async_child_example4();

	// The two main reasons to use groups are
	//	1. Being able to terminate child processes of the child process.
	//	2. Grouping several processes into one, just so they can be terminated at once.
	//local::process_group1();
	//local::process_group2();

	// Boost.Process provides access to the environment of the current process and allows setting it for the child processes.
	//local::environment();
}
