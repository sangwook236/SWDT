#if defined(WIN32)
#define _WIN32_WINNT_NT4 0x0400  // Windows NT 4.0
#define _WIN32_WINNT_WIN2K 0x0500  // Windows 2000
#define _WIN32_WINNT_WINXP 0x0501  // Windows XP
#define _WIN32_WINNT_WIN7 0x0601  // Windows 7
#define _WIN32_WINNT_WIN10 0x0A00  // Windows 10
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#endif

#include <boost/signals2/signal.hpp>
#include <boost/signals2/shared_connection_block.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <vector>
#include <cassert>


namespace {
namespace local {

struct HelloWorld
{
	void operator()() const
	{
		std::cout << "Hello World !!!" << std::endl;
	}
};

struct GoodMorning
{
	void operator()() const
	{
		std::cout << "Good Morning !!!" << std::endl;
	}
};

void print_args(float x, float y)
{
	std::cout << "The arguments are " << x << " and " << y << std::endl;
}

void print_sum(float x, float y)
{
	std::cout << "The sum is " << (x + y) << std::endl;
}

void print_product(float x, float y)
{
	std::cout << "The product is " << (x * y) << std::endl;
}

void print_difference(float x, float y)
{
	std::cout << "The difference is " << (x - y) << std::endl;
}

void print_quotient(float x, float y)
{
	std::cout << "The quotient is " << (x / y) << std::endl;
}

struct Adder
{
	float operator()(float x, float y) const
	{
		//std::cout << "The sum is " << x+y << std::endl;
		return x + y;
	}
};

struct Subtracter
{
	float operator()(float x, float y) const
	{
		//std::cout << "The difference is " << x-y << std::endl;
		return x - y;
	}
};

struct Multiplier
{
	float operator()(float x, float y) const
	{
		//std::cout << "The product is " << x*y << std::endl;
		return x * y;
	}
};

struct Divisor
{
	float operator()(float x, float y) const
	{
		//std::cout << "The quotient is " << x/y << std::endl;
		return x / y;
	}
};

float product(float x, float y) { return x * y; }
float quotient(float x, float y) { return x / y; }
float sum(float x, float y) { return x + y; }
float difference(float x, float y) { return x - y; }

// combiner which returns the maximum value returned by all slots.
template<typename T>
struct maximum
{
	typedef T result_type;

	template<typename InputIterator>
	T operator()(InputIterator first, InputIterator last) const
	{
		// If there are no slots to call, just return the default-constructed value.
		if (first == last) return T();
		T max_value = *first++;
		while (first != last)
		{
			if (max_value < *first)
				max_value = *first;
			++first;
		}

		return max_value;
	}
};

// aggregate_values is a combiner which places all the values returned from slots into a container.
template<typename Container>
struct aggregate_values
{
	typedef Container result_type;

	template<typename InputIterator>
	Container operator()(InputIterator first, InputIterator last) const
	{
		Container values;

		while (first != last)
		{
			values.push_back(*first);
			++first;
		}
		return values;
	}
};

void calling_and_passing_value()
{
	//
	{
		boost::signals2::signal<void(float, float)> sig1;

		sig1.connect(&print_sum);
		sig1.connect(&print_product);

		/*
				typedef boost::function<void (float, float)> signature_type;
				//typedef (void (float, float)) signature_type;  // Oops !!! compile-time error.
				boost::signals2::signal<signature_type> sig2;
		*/

		typedef boost::signals2::signal<void(float, float)> publisher_type;
		typedef publisher_type::slot_function_type subscriber_type;

		publisher_type sig3;
		subscriber_type func1 = &print_difference;
		subscriber_type func2 = &print_quotient;

		sig3.connect(func1);
		sig3.connect(func2);

		sig1(5, 3);
		sig3(2, 1);
	}

	// ordering slot call groups
	{
		boost::signals2::signal<void(float, float)> sig;

		sig.connect(1, &print_args);
		sig.connect(2, &print_sum);
		sig.connect(4, &print_product);
		sig.connect(3, &print_difference);
		sig.connect(5, &print_quotient);

		sig(5.f, 3.f);
	}

	// signal return values #1
	{
		boost::signals2::signal<float(float, float)> sig;

		sig.connect(Adder());
		sig.connect(Subtracter());
		sig.connect(Multiplier());
		sig.connect(Divisor());

		std::cout << sig(5, 3) << std::endl;
	}

	// signal return values #2
	{
		boost::signals2::signal<float(float, float), maximum<float> > sig;

		sig.connect(&product);
		sig.connect(&quotient);
		sig.connect(&sum);
		sig.connect(&difference);

		// Outputs the maximum value returned by the connected slots, in this case 15 from the product function.
		std::cout << "maximum: " << sig(5, 3) << std::endl;
	}

	// signal return values #3
	{
		boost::signals2::signal<float(float, float), aggregate_values<std::vector<float> > > sig;

		sig.connect(&quotient);
		sig.connect(&product);
		sig.connect(&sum);
		sig.connect(&difference);

		const std::vector<float> results = sig(5, 3);
		std::cout << "aggregate values: ";
		std::copy(results.begin(), results.end(), std::ostream_iterator<float>(std::cout, " "));
		std::cout << std::endl;
	}
}

struct NewsMessageArea : public boost::signals2::trackable
{
	void displayNews(const std::string &news) const
	{
		// do something
	}
};

void connention_management()
{
	// disconnecting slots #1
	{
		boost::signals2::signal<void()> sig;

		boost::signals2::connection c(sig.connect(HelloWorld()));
		if (c.connected())
		{
			sig();
			c.disconnect();
		}

		sig();  // do nothing
	}

	// disconnecting slots #2
	{
		boost::signals2::signal<void(float, float)> sig;

		sig.connect(&print_sum);
		sig.connect(&print_difference);
		sig(3, 2);

		sig.disconnect(&print_difference);

		//boost::signals2::signal<void (float, float)>::slot_function_type func = &print_difference;
		boost::function<void(float, float)> func = &print_difference;
		//sig.disconnect(func);  // compile-time error
		sig(3, 2);
	}

	// blocking slots
	{
		boost::signals2::signal<void()> sig;

		boost::signals2::connection c1(sig.connect(HelloWorld()));
		boost::signals2::connection c2(sig.connect(GoodMorning()));

		{
			boost::signals2::shared_connection_block blocker(c1);
			sig();
		}

		sig();
	}

	// scoped connections
	{
		boost::signals2::signal<void()> sig;

		{
			boost::signals2::scoped_connection c1(sig.connect(HelloWorld()));
			boost::signals2::connection c2(sig.connect(GoodMorning()));

			sig();
		}

		sig();
	}

	// automatic connection management
	{
		boost::signals2::signal<void(const std::string &)> deliverNews;

		NewsMessageArea *newsMessageArea = new NewsMessageArea();
		deliverNews.connect(boost::bind(&NewsMessageArea::displayNews, newsMessageArea, _1));
	}

	//typedef boost::signals2::signal<void ()> signal_type;
	//typedef boost::signals2::signal<void ()>::slot_type slot_type;
	//typedef boost::signals2::signal<void ()>::slot_function_type slot_function_type;
	//typedef boost::signals2::connection connection_type;
}

struct MyClass
{
public:
	void doSomething()
	{
		service_.post(boost::bind(&MyClass::doSomethingOp, this));
	}

	void loop()
	{
		service_.run();
	}

private:
	void doSomethingOp() const
	{
		std::cout << "do something ..." << std::endl;
	}

private:
	boost::asio::io_service service_;
};

void signal_in_thread()
{
	boost::signals2::signal<void()> mySignal;

	MyClass myClass;
	mySignal.connect(boost::bind(&MyClass::doSomething, boost::ref(myClass)));

	// launches a thread and executes myClass.loop() there.
	std::cout << "start a thread ..." << std::endl;
	boost::thread thrd(boost::bind(&MyClass::loop, boost::ref(myClass)));

	// calls myClass.doSomething() in this thread, but loop() executes it in the other.
	std::cout << "send a signal to the other threads ..." << std::endl;
	mySignal();

	thrd.join();
}

}  // namespace local
}  // unnamed namespace

void signals_slots()
{
	local::calling_and_passing_value();
	local::connention_management();

	// Use signal in a thread.
	local::signal_in_thread();
}
