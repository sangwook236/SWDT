#include <boost/timer/timer.hpp>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void process()
{
	std::sqrt(123.456L);  // burn some time.
	std::log(123.456L);  // burn some time.
	std::exp(123.456L);  // burn some time.
	std::cos(123.456L);  // burn some time.
	std::sin(123.456L);  // burn some time.
}

}  // namespace local
}  // unnamed namespace

void cpu_timer()
{
	long long num_processing = 100000000LL;

	// cpu_timer.
	{
		//const boost::timer::nanosecond_type twenty_seconds(20 * 1000000000LL);  // 20 [sec].
		const boost::timer::nanosecond_type two_seconds(2 * 1000000000LL);  // 2 [sec].
		boost::timer::nanosecond_type last(0);
		boost::timer::cpu_timer timer;

		bool more_transactions = true;
		while (more_transactions)
		{
			local::process();

			const boost::timer::cpu_times elapsed_times(timer.elapsed());
			const boost::timer::nanosecond_type elapsed(elapsed_times.system + elapsed_times.user);
			if (elapsed >= two_seconds)
			{
				last = elapsed;
				more_transactions = false;
			}
		}

		std::cout << "last: " << last << std::endl;
		const boost::timer::cpu_times elapsed_times(timer.elapsed());
		std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << " sec" << std::endl;
		std::cout << timer.format() << std::endl;
	}

	// auto_cpu_timer.
	{
		boost::timer::auto_cpu_timer timer;

		for (long i = 0; i < num_processing; ++i)
			local::process();
	}
}
