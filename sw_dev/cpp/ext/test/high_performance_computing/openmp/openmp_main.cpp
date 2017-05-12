//#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/cstdint.hpp>
#include <boost/tr1/cmath.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <omp.h>
#if defined(_WIN64) || defined(_WIN32)
#include <xmmintrin.h>
#include <windows.h>
#endif
#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#include <math.h>
#endif


#if defined(_WIN64) || defined(_WIN32)
#if defined(max)
#undef max
#endif
#endif


namespace {
namespace local {

#if defined(_WIN64) || defined(_WIN32)
class Timer
{
public:
	Timer()
	{
		QueryPerformanceFrequency(&freq_);
		QueryPerformanceCounter(&startTime_);
	}
	~Timer()
	{
	}

	__int64 getElapsedTime() const  // [msec].
	{
		LARGE_INTEGER endTime;
		QueryPerformanceCounter(&endTime);
		return (endTime.QuadPart - startTime_.QuadPart) * 1000 / freq_.QuadPart;
	}

private:
	LARGE_INTEGER freq_;
	LARGE_INTEGER startTime_;
};
#endif

void set_value(double &val, const int n)
{
	std::cout << "Set val #" << n << std::endl;
	val = n;
}

void sum_value(const double val, double sum, const int n)
{
	std::cout << "Sum val #" << n << std::endl;
	sum += val;
}

// REF [site] >> http://bisqwit.iki.fi/story/howto/openmp/
//	"Guide into OpenMP: Easy multithreading programming for C++".
void basic()
{
#if defined(__GNUC__)
	{
		const int size = 256;
		double sinTable[size];
#pragma omp simd
		for (int n = 0; n < size; ++n)
			sinTable[n] = std::sin(2 * M_PI * n / size);
	}

	{
		const int size = 256;
		double sinTable[size];
#pragma omp target teams distribute parallel for map(from:sinTable[0:256])
		for (int n = 0; n < size; ++n)
			sinTable[n] = std::sin(2 * M_PI * n / size);
	}
#endif

	std::cout << "----------------------------------------------------------" << std::endl;
	{
#pragma omp parallel
		{
			std::cout << "Hello!" << std::endl;
		}
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
		const int parallelism_enabled = 1;
#pragma omp parallel for if(parallelism_enabled)
		for (int c = 0; c < 10; ++c)
			std::cout << "c = " << c << std::endl;
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
#pragma omp parallel
		{
#pragma omp for
			for (int n = 0; n < 10; ++n)
				std::cout << ' ' << n;
			std::cout << std::endl;
		}
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
#pragma omp parallel num_threads(3)
		{
#pragma omp for
			for (int n = 0; n < 10; ++n)
				std::cout << ' ' << n;
			std::cout << std::endl;
		}
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
		int n;
#pragma omp parallel for private(n)
		for (n = 0; n < 10; ++n)
			std::cout << ' ' << n;
		std::cout << std::endl;
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
		//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for schedule(dynamic, 3)
		for (int n = 0; n < 10; ++n)
			std::cout << ' ' << n;
		std::cout << std::endl;
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
		double val[10] = { 0.0, };
		double sum = 0.0;
#pragma omp parallel for ordered schedule(dynamic)
		for (int n = 0; n < 10; ++n)
		{
			set_value(val[n], n);

#pragma omp ordered
			sum_value(val[n], sum, n);
		}
	}

	std::cout << "----------------------------------------------------------" << std::endl;
	{
//#pragma omp sections
#pragma omp parallel sections
		{
			{
				std::cout << "Work #1." << std::endl;
			}
#pragma omp section
			{
				std::cout << "Work #2." << std::endl;
				std::cout << "Work #3." << std::endl;
			}
#pragma omp section
			{
				std::cout << "Work #4." << std::endl;
			}
		}
	}
}

void parallel_directive()
{
	int nthreads, tid;

	// Fork a team of threads giving them their own copies of variables.
#pragma omp parallel private(tid)
	{
		// Obtain and print thread id.
		tid = omp_get_thread_num();
		std::cout << "Hello World from thread = " << tid << std::endl;

		// Only master thread does this.
		if (0 == tid) 
		{
			nthreads = omp_get_num_threads();
			std::cout << "Number of threads = " << nthreads << std::endl;
		}

	}  // All threads join master thread and terminate.
}

void do_directive()
{
	const int CHUNKSIZE = 100;
	const int N = 60000;
	const int Nj = 100;

	int i, j, chunk;
	float a[N], b[N], c[N], d[N];

	// Some initializations.
	for (i = 0; i < N; ++i)
		a[i] = b[i] = i * 1.0f;
	chunk = CHUNKSIZE;
/*
	{
		boost::int64_t loop_count = 0;
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
		Timer timer;

		{
			for (i = 0; i < N; ++i)
			{
				for (j = 0; j < Nj; ++j)
				{
					c[i] = a[i] + b[i];
					const double beta = std::tr1::beta(100, 100);
					const double err = std::tr1::erf(100);
					const double gamma = std::tr1::lgamma(100);
					d[i] = beta + err + gamma;

					++loop_count;
				}
			}
		}

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
		std::cout << timer.getElapsedTime() << std::endl;
		std::cout << loop_count << std::endl;
	}
*/
	{
		boost::int64_t loop_count = 0;
		std::ostringstream sstream;
		const boost::posix_time::ptime stime = boost::posix_time::microsec_clock::universal_time();
#if defined(_WIN64) || defined(_WIN32)
		Timer timer;
#endif

#pragma omp parallel shared(a, b, c, d, loop_count, chunk) private(i)
//#pragma omp parallel shared(a, b, c, d, loop_count, chunk, i)
		{
#pragma omp for schedule(dynamic, chunk) nowait
			for (i = 0; i < N; ++i)
			{
				//for (j = 0; j < Nj; ++j)
				{
					c[i] = a[i] + b[i];
					const double beta = std::tr1::beta(100, 100);
					const double err = std::tr1::erf(100);
					const double gamma = std::tr1::lgamma(100);
					d[i] = float(beta + err + gamma);

					++loop_count;

					if (0 == i % 1000)
					{
						const int tid = omp_get_thread_num();
						sstream << tid << ": " << i << ", " << a[i] << ", " << b[i] << ", " << c[i] << ", " << loop_count << std::endl;
						std::cout << sstream.str();
					}
				}
			}
		}  // End of parallel section.

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
#if defined(_WIN64) || defined(_WIN32)
		std::cout << timer.getElapsedTime() << std::endl;
#endif
		std::cout << loop_count << std::endl;
	}

	//std::copy(c, c + N, std::ostream_iterator<int>(std::cout, " "));
}

}  // namespace local
}  // unnamed namespace

namespace my_openmp {

}  // namespace my_openmp

int openmp_main(int argc, char *argv[])
{
#if defined(_OPENMP)
#if _OPENMP <= 199810  // Ver. 1.0.
	std::cout << "OpenMP ver. 1.0 supported." << std::endl;
#elif _OPENMP <= 200203  // Ver. 2.0.
	std::cout << "OpenMP ver. 2.0 supported." << std::endl;
#elif _OPENMP <= 200505  // Ver. 2.5.
	std::cout << "OpenMP ver. 2.5 supported." << std::endl;
#elif _OPENMP <= 200805  // Ver. 3.0.
	std::cout << "OpenMP ver. 3.0 supported." << std::endl;
#elif _OPENMP <= 201107  // Ver. 3.1.
	std::cout << "OpenMP ver. 3.1 supported." << std::endl;
#elif _OPENMP <= 201307  // Ver. 4.0.
	std::cout << "OpenMP ver. 4.0 supported." << std::endl;
#elif _OPENMP <= 201511  // Ver. 4.5.
	std::cout << "OpenMP ver. 4.5 supported." << std::endl;
#else
	std::cout << "OpenMP " << _OPENMP << " supported." << std::endl;
#endif
#else
#error ERROR: OpenMP not supported.
#endif

	// Runtime Library Routines.
	{
//#pragma omp parallel
		{
#if _OPENMP >= 201307  // Ver. 4.0.
			std::cout << "#devices = " << omp_get_num_devices() << std::endl;
			std::cout << "Default device ID = " << omp_get_default_device() << std::endl;
			std::cout << (omp_is_initial_device() ? "The current task is executing on the host device" : "The current task is not executing on the host device") << std::endl;
			omp_set_default_device(std::max(omp_get_num_devices() - 2, 0));
			std::cout << "#devices = " << omp_get_num_devices() << std::endl;

			std::cout << "#teams in the current teams region = " << omp_get_num_teams() << std::endl;
			std::cout << "Team ID = " << omp_get_team_num() << std::endl;
#endif

			std::cout << "#max threads = " << omp_get_max_threads() << std::endl;
			std::cout << "#threads in the current team = " << omp_get_num_threads() << std::endl;
			std::cout << "Thread ID = " << omp_get_thread_num() << std::endl;
			omp_set_num_threads(std::max(omp_get_num_threads() - 2, 1));
			std::cout << "#threads in the current team = " << omp_get_num_threads() << std::endl;
		}
	}

	local::basic();

	//local::parallel_directive();
	//local::do_directive();

	return 0;
}
