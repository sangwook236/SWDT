#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include <stdexcept>


namespace {
namespace local {

void normal_task_function(const int id, const int secs)
{
	std::cout << "Start the " << id << "-th task, which is running for " << secs << " seconds." << std::endl;

	std::this_thread::sleep_for(std::chrono::seconds(secs));

	std::cout << "Terminate the " << id << "-th task." << std::endl;
}

void exception_task_function(const int id, const int secs)
{
	std::cout << "Start the " << id << "-th exception task, which is running for " << secs << " seconds." << std::endl;

	std::this_thread::sleep_for(std::chrono::seconds(secs));
	throw std::runtime_error("Exception in exception_task_function()");

	std::cout << "Terminate the " << id << "-th exception task." << std::endl;
}

class add_task_thread
{
public:
	add_task_thread(boost::asio::io_service &io_service)
	: io_service_(io_service)
	{}

public:
	void operator()() const
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));

#if 0
		std::cout << "Requested the io_service to dispatch a handler (#100)." << std::endl;
		io_service_.dispatch(std::bind(normal_task_function, 100, 2));
#else
		std::cout << "Requested the io_service to post a handler (#100)." << std::endl;
		io_service_.post(std::bind(normal_task_function, 100, 2));
#endif
	}

private:
	boost::asio::io_service &io_service_;
};

void basic()
{
	boost::asio::io_service io_service;

	try
	{
		std::cout << "Started an io_service." << std::endl;

#if 0
		std::cout << "Requested the io_service to dispatch given handlers (#1 & #2)." << std::endl;
		io_service.dispatch(std::bind(normal_task_function, 1, 3));
		io_service.dispatch(std::bind(normal_task_function, 2, 1));
		//io_service.dispatch(std::bind(exception_task_function, 3, 1));
#else
		std::cout << "Requested the io_service to post given handlers (#1 & #2)." << std::endl;
		io_service.post(std::bind(normal_task_function, 1, 3));
		io_service.post(std::bind(normal_task_function, 2, 1));
		//io_service.post(std::bind(exception_task_function, 3, 1));
#endif

		std::this_thread::sleep_for(std::chrono::seconds(1));

#if 0
		std::cout << "Run the io_service." << std::endl;
		// Run the io_service object's event processing loop.
		//io_service.run();
		// Run the io_service object's event processing loop to execute at most one handler.
		io_service.run_one();
#else
		std::cout << "Polled the io_service." << std::endl;
		// Run the io_service object's event processing loop to execute ready handlers.
		io_service.poll();
		// Run the io_service object's event processing loop to execute one ready handler.
		//io_service.poll_one();
#endif

		std::cout << "Stopped the io_service." << std::endl;
		// Stop the io_service object's event processing loop.
		//	This function does not block, but instead simply signals the io_service to stop.
		//	Subsequent calls to run(), run_one(), poll() or poll_one() will return immediately until reset() is called.
		io_service.stop();

		while (!io_service.stopped());

#if 0
		std::cout << "Requested the io_service to dispatch a handler (#10)." << std::endl;
		io_service.dispatch(std::bind(normal_task_function, 10, 2));
#else
		std::cout << "Requested the io_service to post a handler (#10)." << std::endl;
		io_service.post(std::bind(normal_task_function, 10, 2));
#endif

		std::cout << "Reset the io_service." << std::endl;
		// Reset the io_service in preparation for a subsequent run() invocation.
		//	This function must be called prior to any second or later set of invocations of the run(), run_one(), poll() or poll_one() functions
		//		when a previous invocation of these functions returned due to the io_service being stopped or running out of work.
		//	After a call to reset(), the io_service object's stopped() function will return false.
		io_service.reset();

		std::unique_ptr<std::thread> thrd(new std::thread(add_task_thread(io_service)));

#if 1
		std::cout << "Re-run the io_service." << std::endl;
		io_service.run();
		//io_service.run_one();
#else
		std::cout << "Re-polled the io_service." << std::endl;
		io_service.poll();
		//io_service.poll_one();
#endif

		thrd->join();

		std::cout << "Exited the io_service normally." << std::endl;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "An exception caught: " << ex.what() << std::endl;

		io_service.stop();
		std::cout << "Exited the io_service exceptionally." << std::endl;
	}
}

class stop_ioservice_thread
{
public:
	stop_ioservice_thread(boost::asio::io_service &io_service, const int secs)
		: io_service_(io_service), secs_(secs)
	{}

public:
	void operator()() const
	{
		std::cout << "Stop the io_service in " << secs_ << " seconds." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(secs_));

		// To effect a shutdown, the application will then need to call the io_service object's stop() member function.
		// This will cause the io_service run() call to return as soon as possible, abandoning unfinished operations and without permitting ready handlers to be dispatched.
		io_service_.stop();  // Allow run() to exit.
	}

private:
	boost::asio::io_service &io_service_;
	const int secs_;
};

class stop_ioservice_work_thread
{
public:
	stop_ioservice_work_thread(std::unique_ptr<boost::asio::io_service::work> &work, const int secs)
		: work_(work), secs_(secs)
	{}

public:
	void operator()() const
	{
		std::cout << "Stop the io_service in " << secs_ << " seconds." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(secs_));

		// If the application requires that all operations and handlers be allowed to finish normally, the work object may be explicitly destroyed.
		work_.reset();  // Allow run() to exit.
	}

private:
	std::unique_ptr<boost::asio::io_service::work> &work_;
	const int secs_;
};

void work_1()
{
	boost::asio::io_service io_service;
	boost::asio::io_service::work work(io_service);

	try
	{
		std::cout << "Started an io_service." << std::endl;
		std::unique_ptr<std::thread> thrd(new std::thread(stop_ioservice_thread(io_service, 3)));

		io_service.run();

		thrd->join();
		std::cout << "Exited the io_service normally." << std::endl;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "An exception caught: " << ex.what() << std::endl;

		io_service.stop();
		std::cout << "Exited the io_service exceptionally." << std::endl;
	}
}

void work_2()
{
	boost::asio::io_service io_service;
	std::unique_ptr<boost::asio::io_service::work> work(new boost::asio::io_service::work(io_service));

	try
	{
		std::cout << "Started an io_service." << std::endl;
		std::unique_ptr<std::thread> thrd(new std::thread(stop_ioservice_work_thread(work, 3)));

		io_service.run();

		thrd->join();
		std::cout << "Exited the io_service normally." << std::endl;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "An exception caught: " << ex.what() << std::endl;

		io_service.stop();
		std::cout << "Exited the io_service exceptionally." << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_io_service()
{
	//local::basic();

	// Prevent io_service's run() call from returning when there is no more work to do.
	local::work_1();
	local::work_2();
}
