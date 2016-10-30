#include <iostream>
#include <thread>
#include <atomic>
#include <future>
#include <vector>
#include <chrono>


namespace {
namespace local {

void async()
{
	// NOTICE [caution] >>
	//	If one does not get the return value of std::async, std::async will wait until its job is terminated.
	{
		// Launch in a separate thread if possible.
		std::async([]() { std::cout << "Print in async." << std::endl; });
		//std::async(std::launch::async | std::launch::deferred, []() { std::cout << "Print in async." << std::endl; });

		std::cout << "Print in the main thread." << std::endl;
	}

	{
		// Launch in a separate thread if possible.
		std::future<void> result(std::async([]() { std::cout << "Print in async." << std::endl; }));
		//std::future<void> result(std::async(std::launch::async | std::launch::deferred, []() { std::cout << "Print in async." << std::endl; }));

		std::cout << "Print in the main thread." << std::endl;
		result.get();
		//std::cout << result.get() << std::endl;  // Error because of void in std::future<void>.
	}

	{
		// Launch in a separate thread if possible.
		std::future<int> result(std::async([](const int lhs, const int rhs) { return lhs + rhs; }, 2, 4));

		std::cout << "Print in the main thread." << std::endl;
		std::cout << result.get() << std::endl;
	}

	{
		std::vector<std::future<int>> futures;
		for (int i = 0; i < 10; ++i)
			futures.push_back(std::async([](const int arg) { return 2 * arg; }, i));

		std::cout << "Print in the main thread." << std::endl;
		for (auto &result : futures)
			std::cout << result.get() << ", ";
		std::cout << std::endl;
	}

	{
		std::atomic_bool done = false;

		auto ret1 = std::async([&done]() {
			std::cout << "Starting the first async." << std::endl;
			done = true;
			std::cout << "Waiting for 1 seconds in the first async." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));
			//done = true;
			std::cout << "Exiting the first async." << std::endl;
		});
		auto ret2 = std::async([&done]() {
			std::cout << "Starting the second async." << std::endl;
			std::cout << "Waiting for 3 seconds in the second async." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(3));
			std::cout << (done ? "Done." : "Still not done.") << std::endl;
			std::cout << "Exiting the second async." << std::endl;
		});
	}
}

void foo()
{
	std::cout << "Print in foo" << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(1));
}

void independent_thread()
{
	std::cout << "Starting concurrent thread." << std::endl;
	std::cout << "Waiting for 2 seconds in the concurrent thread." << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << "Exiting concurrent thread." << std::endl;
}

void thread_caller()
{
	std::cout << "Starting thread caller." << std::endl;
	std::thread thrd(independent_thread);
	thrd.detach();
	std::cout << "Waiting for 1 second in the thread caller." << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(1));
	std::cout << "Exiting thread caller." << std::endl;

	//thrd.join();  // Error: detached thread is not joinable.
}

void thread()
{
	{
		std::thread t1(foo);
		std::thread::id t1_id = t1.get_id();

		std::thread t2(foo);
		std::thread::id t2_id = t2.get_id();

		std::cout << "t1's id: " << t1_id << std::endl;
		std::cout << "t2's id: " << t2_id << std::endl;

		t1.join();
		t2.join();
	}

	{
		thread_caller();
		std::cout << "Waiting for 5 seconds in the main thread." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
}

}  // namespace local
}  // unnamed namespace

void thread()
{
	local::async();
	//local::thread();
}
