#define BOOST_CHRONO_DONT_PROVIDES_DEPRECATED_IO_SINCE_V2_0_0 1
#define  BOOST_CHRONO_HEADER_ONLY 1

#include <boost/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>
#include <boost/chrono/floor.hpp>
#include <boost/chrono/round.hpp>
#include <boost/chrono/ceil.hpp>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>


namespace {
namespace local {

void process()
{
	std::sqrt(123.456L);  // burn some time
	std::log(123.456L);  // burn some time
	std::exp(123.456L);  // burn some time
	std::cos(123.456L);  // burn some time
	std::sin(123.456L);  // burn some time
}

void duration()
{
	{
		typedef boost::chrono::duration<long, boost::ratio<60> > minutes_type;
		minutes_type m1(3);
		minutes_type m2(2);
		minutes_type m3 = m1 + m2;

		typedef boost::chrono::duration<long long, boost::micro> microseconds_type;
		microseconds_type us1(3);
		microseconds_type us2(2);
		microseconds_type us3 = us1 + us2;

		microseconds_type us4 = m3 + us3;   // us4 stores 300000005
		std::cout << "micro-seconds: " << us4 << std::endl;
		std::cout << "tick counts: " << us4.count() << std::endl;

		minutes_type m4 = boost::chrono::duration_cast<minutes_type>(m3 + us3);
		std::cout << "minutes: " << m4 << std::endl;
		std::cout << "tick counts: " << m4.count() << std::endl;

		typedef boost::chrono::duration<double, boost::ratio<60> > dminutes_type;
		dminutes_type dm4 = m3 + us3;  // dm4.count() == 5.000000083333333
		std::cout << "dminutes: " << std::setprecision(16) << dm4 << std::endl;
		std::cout << "tick counts: " << std::setprecision(16) << dm4.count() << std::endl;
	}

	{
		boost::chrono::minutes m1(3);
		boost::chrono::minutes m2(2);
		boost::chrono::minutes m3 = m1 + m2;

		boost::chrono::microseconds us1(3);
		boost::chrono::microseconds us2(2);
		boost::chrono::microseconds us3 = us1 + us2;

		boost::chrono::microseconds us4 = m3 + us3;   // us4 stores 300000005
		std::cout << "micro-seconds: " << us4 << std::endl;
		std::cout << "tick counts: " << us4.count() << std::endl;

		boost::chrono::minutes m4 = boost::chrono::duration_cast<boost::chrono::minutes>(m3 + us3);
		std::cout << "minutes: " << m4 << std::endl;
		std::cout << "tick counts: " << m4.count() << std::endl;

		typedef boost::chrono::duration<double, boost::ratio<60> > dminutes_type;
		dminutes_type dm4 = m3 + us3;  // dm4.count() == 5.000000083333333
		std::cout << "dminutes: " << std::setprecision(16) << dm4 << std::endl;
		std::cout << "tick counts: " << std::setprecision(16) << dm4.count() << std::endl;
	}

	{
		boost::chrono::nanoseconds start;
		{
			long long num_processing = 100000000LL;
			for (long i = 0; i < num_processing; ++i)
				local::process();
		}
		boost::chrono::nanoseconds end;

		boost::chrono::milliseconds d = boost::chrono::duration_cast<boost::chrono::milliseconds>(end - start);

		// d now holds the number of milliseconds from start to end.

		std::cout << d.count() << "ms" << std::endl;

		boost::chrono::nanoseconds d1 = end - start;
		std::cout << d1.count() << "ns" << std::endl;

		boost::chrono::duration<double> d2 = end - start;
		std::cout << d2.count() << "s" << std::endl;
	}

	{
		boost::chrono::milliseconds ms(2500);
		std::cout << boost::chrono::floor<boost::chrono::seconds>(ms) << std::endl;
		std::cout << boost::chrono::round<boost::chrono::seconds>(ms) << std::endl;
		std::cout << boost::chrono::ceil<boost::chrono::seconds>(ms) << std::endl;

		typedef boost::chrono::duration<long, boost::ratio<1, 30> > frame_rate_type;
		ms = boost::chrono::milliseconds(2516);

		std::cout << boost::chrono::floor<frame_rate_type>(ms) << std::endl;
		std::cout << boost::chrono::round<frame_rate_type>(ms) << std::endl;
		std::cout << boost::chrono::ceil<frame_rate_type>(ms) << std::endl;
	}
}

void clocks()
{
	// system_clock:
	// system_clock is useful when you need to correlate the time with a known epoch so you can convert it to a calendar time

	// steady_clock:
	// steady_clock is useful when you need to wait for a specific amount of time.
	// steady_clock time can not be reset.
	// As other steady clocks, it is usually based on the processor tick.
	{
#ifdef BOOST_CHRONO_HAS_CLOCK_STEADY
		boost::chrono::steady_clock::time_point start = boost::chrono::steady_clock::now();
		boost::chrono::steady_clock::duration delay = boost::chrono::seconds(2);

		std::cout << "waiting for 2 seconds ..." << std::endl;
		while (boost::chrono::steady_clock::now() - start <= delay)
		{
			// do nothing
		}
#else
		std::cout << "steady_clock is not supported ..." << std::endl;
#endif
	}

	// high_resolution_clock
	// When available, high_resolution_clock is usually more expensive than the other system-wide clocks,
	// so they are used only when the provided resolution is required to the application.

	// process_cpu_clock:
	// Process and thread clocks are used usually to measure the time spent by code blocks, as a basic time-spent profiling of different blocks of code

	// thread_clock:
	// You can use thread_clock whenever you want to measure the time spent by the current thread
	{
		boost::chrono::thread_clock::time_point start = boost::chrono::thread_clock::now();

		// do something
		{
			long long num_processing = 100000000LL;
			for (long i = 0; i < num_processing; ++i)
				local::process();
		}

		boost::chrono::thread_clock::time_point end = boost::chrono::thread_clock::now();

		boost::chrono::milliseconds d1 = boost::chrono::duration_cast<boost::chrono::milliseconds>(end - start);
		std::cout << d1.count() << "ms" << std::endl;

		typedef boost::chrono::duration<double> dseconds_type;  // seconds, stored with a double.
		dseconds_type d2 = end - start;
		std::cout << std::setprecision(16) << d2.count() << "s" << std::endl;
	}
}

void time_point()
{
	{
		boost::chrono::steady_clock::time_point start = boost::chrono::steady_clock::now();

		{
			long long num_processing = 100000000LL;
			for (long i = 0; i < num_processing; ++i)
				local::process();
		}

		const boost::chrono::duration<double> sec = boost::chrono::steady_clock::now() - start;
		std::cout << "process() took " << sec.count() << " seconds" << std::endl;
	}

	{
		// delay for at least 500 nanoseconds
		auto go = boost::chrono::steady_clock::now() + boost::chrono::nanoseconds(500);

		std::cout << "waiting for 500 nano-seconds ..." << std::endl;
		while (boost::chrono::steady_clock::now() < go)
			/* do nothing */ ;
	}

	{
		typedef boost::chrono::time_point<boost::chrono::steady_clock, boost::chrono::duration<double, boost::ratio<3600> > > T;
		T tp = boost::chrono::steady_clock::now();
		std::cout << tp << std::endl;
	}
}

void io()
{
	typedef boost::chrono::duration<long long, boost::ratio<1, 2500000000> > clock_tick_type;

	{
		std::cout << "milliseconds(1) = " << boost::chrono::milliseconds(1) << std::endl;
		std::cout << "milliseconds(3) + microseconds(10) = " << boost::chrono::milliseconds(3) + boost::chrono::microseconds(10) << std::endl;
		std::cout << "hours(3) + minutes(10) = " << boost::chrono::hours(3) + boost::chrono::minutes(10) << std::endl;

		std::cout << "clock_tick_type(3) + boost::chrono::nanoseconds(10) = " << clock_tick_type(3) + boost::chrono::nanoseconds(10) << std::endl;


		//
		std::cout << "\nSet cout to use short names:" << std::endl;

		std::cout << boost::chrono::symbol_format;
		//std::cout << boost::chrono::name_format;

		std::cout << "milliseconds(3) + microseconds(10) = " << boost::chrono::milliseconds(3) + boost::chrono::microseconds(10) << std::endl;
		std::cout << "hours(3) + minutes(10) = " << boost::chrono::hours(3) + boost::chrono::minutes(10) << std::endl;

		std::cout << "clock_tick_type(3) + nanoseconds(10) = " << clock_tick_type(3) + boost::chrono::nanoseconds(10) << std::endl;
	}

	{
		std::cout << boost::chrono::duration_fmt(boost::chrono::duration_style::symbol);
		//std::cout << boost::chrono::duration_fmt(boost::chrono::duration_style::prefix);

		std::cout << "milliseconds(3) - microseconds(10) = " << boost::chrono::milliseconds(3) - boost::chrono::microseconds(10) << std::endl;
	}

	{
		std::istringstream in("5000 milliseconds 4000 ms 3001 ms");

		boost::chrono::seconds d(0);

		in >> d;
		assert(in.good());
		std::cout << "d == seconds(5): " << std::boolalpha << (d == boost::chrono::seconds(5)) << std::endl;

		in >> d;
		assert(in.good());
		std::cout << "d == seconds(4): " << std::boolalpha << (d == boost::chrono::seconds(4)) << std::endl;

		in >> d;
		assert(in.fail());
		std::cout << "d == seconds(4): " << std::boolalpha << (d == boost::chrono::seconds(4)) << std::endl;
	}

	{
		boost::chrono::high_resolution_clock::time_point t0 = boost::chrono::high_resolution_clock::now();

		std::stringstream io;
		io << t0;

		boost::chrono::high_resolution_clock::time_point t1;
		io >> t1;
		assert(!io.fail());

		std::cout << io.str() << std::endl;
		std::cout << t0 << std::endl;
		std::cout << t1 << std::endl;

		boost::chrono::high_resolution_clock::time_point t = boost::chrono::high_resolution_clock::now();
		std::cout << t << std::endl;
		std::cout << "That took " << t - t0 << std::endl;
		std::cout << "That took " << t - t1 << std::endl;
	}

	{
		// FIXME [check] >> not correctly working

		std::cout << boost::chrono::system_clock::now() << std::endl;

		std::cout << boost::chrono::time_fmt(boost::chrono::timezone::local) << boost::chrono::system_clock::now() << std::endl;
		std::cout << boost::chrono::time_fmt(boost::chrono::timezone::local, "%A %B %e, %Y %r") << boost::chrono::system_clock::now() << std::endl;

		std::cout << boost::chrono::time_fmt(boost::chrono::timezone::utc) << boost::chrono::system_clock::now() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void chrono()
{
	std::cout << "duration --------------------------------" << std::endl;
	local::duration();
	std::cout << "clocks ----------------------------------" << std::endl;
	local::clocks();
	std::cout << "time point-------------------------------" << std::endl;
	local::time_point();
	std::cout << "I/O -------------------------------------" << std::endl;
	local::io();

	//Boost.Chrono.Stopwatch
}
