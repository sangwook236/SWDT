#include <windows.h>
#include <vector>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

void wait(const long msec)
{
	std::cout << "start waiting for " << msec << " milli-seconds ...";

#if defined(WIN32)
	Sleep(msec);
#else
	throw std::runtime_error("not yet supported");
#endif

	std::cout << " end waiting" << std::endl;
}

void pointer_arithmetic_1(const LARGE_INTEGER &freq)
{
	wait(5000);

	//const std::size_t MAX_COUNT = 10000000;
	const std::size_t MAX_COUNT = 10000;

	{
		double data[MAX_COUNT] = { 0.0, };

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		for (std::size_t i = 0; i < MAX_COUNT; ++i)
			data[i] = i;
		QueryPerformanceCounter(&endTime);

		std::cout << ">>>>> using array: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}

	wait(5000);

	{
		double data[MAX_COUNT] = { 0.0, };

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		double *ptr = data;
		for (std::size_t i = 0; i < MAX_COUNT; ++i, ++ptr)
			*ptr = i;
		QueryPerformanceCounter(&endTime);

		std::cout << ">>>>> using pointer: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}
}

void pointer_arithmetic_2(const LARGE_INTEGER &freq)
{
	wait(5000);

	//const std::size_t MAX_COUNT = 10000000;
	const std::size_t MAX_COUNT = 10000;

	{
		double data[MAX_COUNT] = { 0.0, };

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		double *ptr = data;
		for (std::size_t i = 0; i < MAX_COUNT; ++i, ++ptr)
			*ptr = i;
		QueryPerformanceCounter(&endTime);

		std::cout << ">>>>> using pointer: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}
}

void looping_1(const LARGE_INTEGER &freq)
{
	const std::size_t MAX_COUNT = 10000000;

	{
		double *data = new double [MAX_COUNT];

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		double *ptr = data;
		for (std::size_t i = 0; i < MAX_COUNT; ++i, ++ptr)
			*ptr = i;
		QueryPerformanceCounter(&endTime);

		std::cout << ">>>>> using for: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;

		delete [] data;
	}
}

void looping_2(const LARGE_INTEGER &freq)
{
	const std::size_t MAX_COUNT = 10000000;

	{
		double *data = new double [MAX_COUNT];

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		double *ptr = data;
		std::size_t i = 0;
		while (i < MAX_COUNT)
		{
			*(ptr++) = i;
			++i;
		}
		QueryPerformanceCounter(&endTime);

		std::cout << ">>>>> using while: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;

		delete [] data;
	}
}

void pow(const LARGE_INTEGER &freq)
{
	const std::size_t MAX_COUNT = 10000000;

	{
		long double sum = 0.0;

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		for (std::size_t i = 0; i < MAX_COUNT; ++i)
			sum += std::pow(1.01, 4.0);
		QueryPerformanceCounter(&endTime);

		std::cout << "sum = " << sum << std::endl;
		std::cout << ">>>>> using pow: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}

	{
		long double sum = 0.0, tmp;

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		for (std::size_t i = 0; i < MAX_COUNT; ++i)
		{
			//sum += 1.01 * 1.01 * 1.01 * 1.01;
			tmp = 1.01 * 1.01;
			sum += tmp * tmp;
		}
		QueryPerformanceCounter(&endTime);

		std::cout << "sum = " << sum << std::endl;
		std::cout << ">>>>> using multiplication: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}
}

void vector_size(const LARGE_INTEGER &freq)
{
	const std::size_t MAX_COUNT = 10000000;

	{
		long double sum = 0.0;
		std::vector<double> vec(MAX_COUNT, 0.0);
		for (std::size_t i = 0; i < MAX_COUNT; ++i)
			vec[i] = i;

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		for (std::size_t i = 0; i < vec.size(); ++i)
			sum += vec[i];
		QueryPerformanceCounter(&endTime);

		std::cout << "sum = " << sum << std::endl;
		std::cout << ">>>>> using vector::size(): " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}

	{
		long double sum = 0.0;
		std::vector<double> vec(MAX_COUNT, 0.0);
		for (std::size_t i = 0; i < MAX_COUNT; ++i)
			vec[i] = i;

		std::size_t count = vec.size();

		LARGE_INTEGER startTime, endTime;
		QueryPerformanceCounter(&startTime);
		for (std::size_t i = 0; i < count; ++i)
			sum += vec[i];
		QueryPerformanceCounter(&endTime);

		std::cout << "sum = " << sum << std::endl;
		std::cout << ">>>>> using size constant: " << std::setprecision(9) << ((endTime.QuadPart - startTime.QuadPart) * 1000000.0 / freq.QuadPart) << " usec" << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void performance_analysis()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	std::cout << "pointer arithmetic -----------------------------------------------------------" << std::endl;
	local::pointer_arithmetic_1(freq);
	local::pointer_arithmetic_2(freq);
	
	std::cout << "\nlooping ----------------------------------------------------------------------" << std::endl;
	local::looping_1(freq);
	local::looping_2(freq);
	
	std::cout << "\npow() ------------------------------------------------------------------------" << std::endl;
	local::pow(freq);
	
	std::cout << "\nstd::vector::size() ----------------------------------------------------------" << std::endl;
	local::vector_size(freq);
}
