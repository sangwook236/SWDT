#include "stdafx.h"
#include <omp.h>
//#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/cstdint.hpp>
#include <boost/tr1/cmath.hpp>
#include <iterator>
#include <iostream>


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

void parallel_directive();
void do_directive();

int main(int argc, char **argv)
{
	try
	{
		//parallel_directive();
		do_directive();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught !!!" << std::endl;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}

void parallel_directive()
{
	int nthreads, tid;

	// Fork a team of threads giving them their own copies of variables.
#pragma omp parallel private(tid)
	{
		// Obtain and print thread id.
		tid = omp_get_thread_num();
		printf("Hello World from thread = %d\n", tid);

		// Only master thread does this.
		if (tid == 0) 
		{
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
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
		Timer timer;

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
					d[i] = beta + err + gamma;

					++loop_count;
					const int tid = omp_get_thread_num();
					sstream << tid << ": " << i << ", " << a[i] << ", " << b[i] << ", " << c[i] << ", " << loop_count << std::endl;
					std::cout << sstream.str();
				}
			}
		}  // End of parallel section.

		const boost::posix_time::ptime etime = boost::posix_time::microsec_clock::universal_time();
		const boost::posix_time::time_duration td = etime - stime;

		std::cout << stime << " : " << etime << " : " << td << std::endl;
		std::cout << timer.getElapsedTime() << std::endl;
		std::cout << loop_count << std::endl;
	}

	//std::copy(c, c + N, std::ostream_iterator<int>(std::cout, " "));
}
