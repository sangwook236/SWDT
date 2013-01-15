#include <boost/progress.hpp>
#include <iostream>


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

}  // namespace local
}  // unnamed namespace

void progress_timer()
{
	long long num_processing = 100000000LL;

	// timer
	{
		boost::timer timer;  // start timing

		for (long i = 0; i < num_processing; ++i)
			local::process();

		std::cout << "elapsed time: " << timer.elapsed() << std::endl;
	}

	// progress_timer
	{
		boost::progress_timer timer;  // start timing

		for (long i = 0; i < num_processing; ++i)
			local::process();
	}

	// progress_display
	{
		boost::progress_display show_progress((unsigned long)num_processing);
		for (long long i = 0; i < num_processing; ++i)
		{
			local::process();

			++show_progress;
		}
	}
}
