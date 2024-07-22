//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#if defined(_WIN64) || defined(_WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif


namespace my_common_api {

void generate_signal(const double startTime, const double endTime, const double interval, std::vector<double> &signal)
{
	const double frequency1 = 5.0, amplitude1 = 2.0;
	const double frequency2 = 20.0, amplitude2 = 1.0;
	const double frequency3 = 35.0, amplitude3 = 4.0;
	const double frequency4 = 60.0, amplitude4 = 3.0;

	const size_t numSignal = (size_t)std::ceil((endTime - startTime) / interval);
	signal.reserve(numSignal + 1);
	for (double t = startTime; t <= endTime; t += interval)
		signal.push_back(
			amplitude1 * std::sin(2.0 * M_PI * frequency1 * t) +
			amplitude2 * std::sin(2.0 * M_PI * frequency2 * t) +
			amplitude3 * std::sin(2.0 * M_PI * frequency3 * t) +
			amplitude4 * std::sin(2.0 * M_PI * frequency4 * t)
		);
}

void write_signal(const std::vector<double> &signal, const std::string &filepath)
{
	std::ofstream stream(filepath, std::ios::trunc);
	if (!stream)
	{
		std::cerr << "File not found: " << filepath << std::endl;
		return;
	}

#if 0
	// Too slow.
	for (const auto &sig : signal)
		stream << sig << std::endl;
#else
	std::copy(signal.begin(), signal.end(), std::ostream_iterator<double>(stream, "\n"));
#endif
}

// REF [function] >> filter() in ${SWL_HOME}/cpp/src/rnd_util/SignalProcessing.cpp.
void filter(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &x, std::vector<double> &y)
{
	// a[0] * y[n] = b[0] * x[n] + b[1] * x[n - 1] + ... + b[nb] * x[n - nb] - a[1] * y[n - 1] - ... - a[na] * y[n - na].

	const size_t na = a.size();
	const size_t nb = b.size();
	const size_t nx = x.size();

	y.reserve(nx);
	if (na == nb)
	{
		for (size_t n = 0; n < nx; ++n)
		{
			double sum = b[0] * x[n];
			for (size_t i = 1; i <= nb && i <= n; ++i)
				sum += b[i] * x[n - i] - a[i] * y[n - i];

			y.push_back(sum / a[0]);
		}
	}
	else
	{
		for (size_t n = 0; n < nx; ++n)
		{
			double sum = 0.0;
			for (size_t i = 0; i <= nb && i <= n; ++i)
				sum += b[i] * x[n - i];
			for (size_t i = 1; i <= na && i <= n; ++i)
				sum -= a[i] * y[n - i];

			y.push_back(sum / a[0]);
		}
	}
}

}  // namespace my_common_api

int main(int argc, char *argv[])
{
	int fast_bilateral_filter_main(int argc, char *argv[]);
	int nyu_depth_toolbox_v2_main(int argc, char *argv[]);

	int dspfilters_main(int argc, char *argv[]);
	int sigproc_main(int argc, char *argv[]);

	int aquila_main(int argc, char *argv[]);
	int itpp_main(int argc, char *argv[]);
	int spuc_main(int argc, char *argv[]);
	int tspl_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Fast bilateral filter algorithm -------------------------------------" << std::endl;
		//retval = fast_bilateral_filter_main(argc, argv);

		std::cout << "\nNYU Depth Toolbox V2 ------------------------------------------------" << std::endl;
		//retval = nyu_depth_toolbox_v2_main(argc, argv);

		std::cout << "\nDspFilters library --------------------------------------------------" << std::endl;
		//retval = dspfilters_main(argc, argv);

		std::cout << "\nsigproc library -----------------------------------------------------" << std::endl;
		//retval = sigproc_main(argc, argv);

		std::cout << "\nAquila library ------------------------------------------------------" << std::endl;
		retval = aquila_main(argc, argv);

		std::cout << "\nIT++ library --------------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = itpp_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nSignal Processing using C++ (SPUC) library --------------------------" << std::endl;
		//retval = spuc_main(argc, argv);

		std::cout << "\nSignal Processing Library in C++ (tspl) -----------------------------" << std::endl;
		//retval = tspl_main(argc, argv);  // Not yet implemented.
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
