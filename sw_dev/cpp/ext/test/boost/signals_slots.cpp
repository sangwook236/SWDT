#include <boost/signals2/signal.hpp>
#include <boost/signals2/shared_connection_block.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <iostream>
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

void print_sum(float x, float y)
{
	std::cout << "The sum is " << x+y << std::endl;
}

void print_product(float x, float y)
{
	std::cout << "The product is " << x*y << std::endl;
}

void print_difference(float x, float y)
{
	std::cout << "The difference is " << x-y << std::endl;
}

void print_quotient(float x, float y)
{
	std::cout << "The quotient is " << x/y << std::endl;
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

template<typename T>
struct maximum_chooser
{
	typedef T result_type;

	template<typename InputIterator>
	T operator()(InputIterator first, InputIterator last) const
	{
		return *max_element(first, last);
	}
};

template<typename T>
struct result_aggregater
{
	typedef T result_type;

	template<typename InputIterator>
	T operator()(InputIterator first, InputIterator last) const
	{
		return T(first, last);
	}
};

void signals_slots__calling_and_passing_value()
{
	//
	{
		boost::signals2::signal<void (float, float)> sig1;

		sig1.connect(&print_sum);
		sig1.connect(&print_product);

/*
		typedef boost::function<void (float, float)> signature_type;
		//typedef (void (float, float)) signature_type;  // Oops !!! compile-time error
		boost::signals2::signal<signature_type> sig2;
*/

		typedef boost::signals2::signal<void (float, float)> publisher_type;
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
		boost::signals2::signal<void (float, float)> sig;

		sig.connect(1, &print_sum);
		sig.connect(3, &print_product);
		sig.connect(2, &print_difference);
		sig.connect(4, &print_quotient);

		sig(5, 3);
	}

	// signal return values #1
	{
		boost::signals2::signal<float (float, float)> sig;

		sig.connect(Adder());
		sig.connect(Subtracter());
		sig.connect(Multiplier());
		sig.connect(Divisor());

		std::cout << sig(5, 3) << std::endl;
	}

	// signal return values #2
	{
		boost::signals2::signal<float (float, float), maximum_chooser<float> > sig;

		sig.connect(Adder());
		sig.connect(Subtracter());
		sig.connect(Multiplier());
		sig.connect(Divisor());

		std::cout << sig(5, 3) << std::endl;
	}

	// signal return values #3
	{
		boost::signals2::signal<float (float, float), result_aggregater<std::vector<float> > > sig;

		sig.connect(Adder());
		sig.connect(Subtracter());
		sig.connect(Multiplier());
		sig.connect(Divisor());

		const std::vector<float> results = sig(5, 3);
		std::copy(results.begin(), results.end(), std::ostream_iterator<float>(std::cout, " "));
		std::cout << std::endl;
	}
}

struct NewsMessageArea: public boost::signals2::trackable
{
    void displayNews(const std::string &news) const
    {
        // do something
    }
};

void signals_slots__connention_management()
{
	// disconnecting slots #1
	{
		boost::signals2::signal<void ()> sig;

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
		boost::signals2::signal<void (float, float)> sig;

		sig.connect(&print_sum);
		sig.connect(&print_difference);
		sig(3, 2);

		sig.disconnect(&print_difference);

		//boost::signals2::signal<void (float, float)>::slot_function_type func = &print_difference;
		boost::function<void (float, float)> func = &print_difference;
		//sig.disconnect(func);  // compile-time error
		sig(3, 2);
	}

	// blocking slots
	{
		boost::signals2::signal<void ()> sig;

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
		boost::signals2::signal<void ()> sig;

		{
			boost::signals2::scoped_connection c1(sig.connect(HelloWorld()));
			boost::signals2::connection c2(sig.connect(GoodMorning()));

			sig();
		}

		sig();
	}

	// automatic connection management
	{
		boost::signals2::signal<void (const std::string &)> deliverNews;

		NewsMessageArea *newsMessageArea = new NewsMessageArea();
		deliverNews.connect(boost::bind(&NewsMessageArea::displayNews, newsMessageArea, _1));
	}

	//typedef boost::signals2::signal<void ()> signal_type;
	//typedef boost::signals2::signal<void ()>::slot_type slot_type;
	//typedef boost::signals2::signal<void ()>::slot_function_type slot_function_type;
	//typedef boost::signals2::connection connection_type;
}

}  // namespace local
}  // unnamed namespace

void signals_slots()
{
	local::signals_slots__calling_and_passing_value();
	local::signals_slots__connention_management();
}
