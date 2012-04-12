#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

boost::mutex io_mutex; // The iostreams are not guaranteed to be thread-safe!

struct thread_alarm
{
	thread_alarm(const int secs)
	: secs_(secs)
	{
		std::cout << "thread_alarm object is created." << std::endl;
	}
	~thread_alarm()
	{
		std::cout << "thread_alarm object is deleted." << std::endl;
	}

public:
	void operator()()
	{
		//boost::xtime xt;
		//boost::xtime_get(&xt, boost::TIME_UTC);
		//xt.sec += secs_;
		//boost::thread::sleep(xt);
		boost::this_thread::sleep(boost::posix_time::seconds(secs_));
		//boost::this_thread::sleep(boost::posix_time::milliseconds(secs_ * 1000));

		std::cout << "alarm sounded..." << std::endl;
	}

private:
	int secs_;
};

void simple_thread_2_proc(void *param)
{
	if (!param) return;
	const int id = static_cast<int>(*(int*)param);

	{
		boost::mutex::scoped_lock lock(io_mutex);
		std::cout << "thread id: " << id << std::endl;
	}
	//boost::xtime xt;
	//boost::xtime_get(&xt, boost::TIME_UTC);
	//xt.sec += 1;
	//boost::thread::sleep(xt);
	boost::this_thread::sleep(boost::posix_time::seconds(1));
}

class thread_adapter
{
public:
	thread_adapter(void (*func)(void*), void* param)
	: proc_(func), param_(param)
	{}

public:
	void operator()() const
	{
		proc_(param_);
	}

private:
	void (*proc_)(void*);
	void* param_;
};

class counter
{
public:
    counter() : count_(0) { }

    int increment()
	{
        boost::mutex::scoped_lock lock(mutex_);
        return ++count_;
    }

private:
    boost::mutex mutex_;
    int count_;
};

counter c;

void change_count()
{
    int i = c.increment();

    boost::mutex::scoped_lock lock(io_mutex);
    std::cout << "count == " << i << std::endl;
}

void simple_thread_1()
{
	std::cout << "setting alarm for 5 seconds..." << std::endl;

	const int secs = 5;
	//thread_alarm alarm(secs);
	//boost::scoped_ptr<boost::thread> thrd(new boost::thread(alarm));  // create thread
	boost::scoped_ptr<boost::thread> thrd(new boost::thread(thread_alarm(secs)));  // create thread

#if 0
	boost::xtime xt;
	boost::xtime_get(&xt, boost::TIME_UTC);
	xt.sec += secs - 2;
	boost::thread::sleep(xt);
#else
	boost::this_thread::sleep(boost::posix_time::seconds(secs - 2));
#endif

	thrd.reset();  // terminate thread

	if (thrd.get())
	{
		thrd->join();
		std::cout << "thread is joined" << std::endl;
	}
	else
		std::cout << "thread has already been terminated" << std::endl;
}

void simple_thread_2()
{
	const int thrda_id = 1;
	const int thrdb_id = 2;
	boost::thread thrda(thread_adapter(&simple_thread_2_proc, (void*)&thrda_id));
	boost::thread thrdb(&simple_thread_2_proc, (void*)&thrdb_id);
	//boost::thread thrdb(thread_adapter(&simple_thread_2_proc, (void*)&thrdb_id));

#if 0
	boost::xtime xt;
	boost::xtime_get(&xt, boost::TIME_UTC);
	xt.sec += 1;
	boost::thread::sleep(xt);
	//boost::thread::yield();
#else
	boost::this_thread::sleep(boost::posix_time::seconds(1));
	//boost::this_thread::yield();
#endif

	thrda.join();
	thrdb.join();
}

void thread_group()
{
    const int num_threads = 4;

    boost::thread_group thrds;
    for (int i = 0; i < num_threads; ++i)
        thrds.create_thread(&change_count);

    thrds.join_all();
}

}  // namespace local
}  // unnamed namespace

void thread()
{
	local::simple_thread_1();
	std::cout << "***" << std::endl;
	local::simple_thread_2();
	std::cout << "***" << std::endl;
	local::thread_group();
}
