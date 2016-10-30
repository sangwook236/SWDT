//#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/cstdint.hpp>
#include <boost/tr1/cmath.hpp>
#include <iostream>
#include <iterator>
#include <omp.h>
#if defined(_WIN64) || defined(_WIN32)
#include <xmmintrin.h>
#include <windows.h>
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
//#pragma omp parallel shared(a, b, c, d, loop_count, i, chunk)
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

					if (i % 1000 == 0)
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
	local::parallel_directive();
	//local::do_directive();

	return 0;
}
