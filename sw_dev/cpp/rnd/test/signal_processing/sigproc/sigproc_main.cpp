//#include "stdafx.h"
#include "../sigproc_lib/iir.h"
#include <iostream>
#include <vector>
#include <cmath>
#if defined(_WIN64) || defined(_WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif


namespace my_common_api {

void generate_signal(const double startTime, const double endTime, const double interval, std::vector<double> &signal);
void write_signal(const std::vector<double> &signal, const std::string &filepath);
void filter(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &x, std::vector<double> &y);

}  // namespace my_common_api

namespace {
namespace local {

// REF [site] >> http://www.exstrom.com/journal/sigproc/bwlp.c
void butterworth_low_pass_filter_coefficients()
{
	// Signal.
	const double samplingRate = 1000.0;  // [Hz].
	const double startTime = 0.0;  // [sec].
	const double endTime = 2.0;  // [sec].

	std::vector<double> signal;
	my_common_api::generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
	my_common_api::write_signal(signal, "./data/signal_processing/before_butterworth_lowpass_filtering2.txt");

	// Filter.
	const size_t order = 4;  // Filter order.
	const double Fc = 40.0;  // Cutoff frequency [Hz].
	const double Fs = samplingRate;  // Sampling rate [Hz].

	const double fcf = 2.0 * Fc / Fs;  // Cutoff frequency as a fraction of Pi [0, 1].
	const bool scaleFlag = true;  // Scale flag: true to scale, false to not scale ccof.

	// Calculate the d coefficients.
	const double *dcof = dcof_bwlp(order, fcf);  // d coefficients.
	if (nullptr == dcof)
	{
		std::cerr << "Unable to calculate d coefficients" << std::endl;
		return;
	}

	// Calculate the c coefficients.
	const int *ccof = ccof_bwlp(order);  // c coefficients.
	if (nullptr == ccof)
	{
		std::cerr << "Unable to calculate c coefficients" << std::endl;
		return;
	}

	// Output the c coefficients.
	std::cout << "Number of c coefficients (numerator) = " << order + 1 << std::endl;  // Number of c coefficients.
	if (!scaleFlag)
	{
		for (size_t i = 0; i <= order; ++i)
			std::cout << ccof[i] << ", ";
		std::cout << std::endl;
	}
	else
	{
		const double sf = sf_bwlp(order, fcf);  // Scaling factor for the c coefficients.

		for (size_t i = 0; i <= order; ++i)
			std::cout << (double)ccof[i] * sf << ", ";
		std::cout << std::endl;
	}

	// Output the d coefficients.
	std::cout << "Number of d coefficients (denominator) = " << order + 1 << std::endl;;  // Number of d coefficients.
	for (size_t i = 0; i <= order; ++i)
		std::cout << dcof[i] << ", ";
	std::cout << std::endl;

	// Filter signal.
	std::vector<double> filtered_signal;
	my_common_api::filter(std::vector<double>(dcof, dcof + order + 1), std::vector<double>(ccof, ccof + order + 1), signal, filtered_signal);
	my_common_api::write_signal(filtered_signal, "./data/signal_processing/after_butterworth_lowpass_filtering2.txt");

	// Clean-up.
	free((void *)dcof);
	free((void *)ccof);
}

// REF [site] >> http://www.exstrom.com/journal/sigproc/bwbp.c
void butterworth_band_pass_filter_coefficients()
{
	// Signal.
	const double samplingRate = 1000.0;  // [Hz].
	const double startTime = 0.0;  // [sec].
	const double endTime = 2.0;  // [sec].

	std::vector<double> signal;
	my_common_api::generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
	//my_common_api::write_signal(signal, "./data/signal_processing/before_butterworth_bandpass_filtering2.txt");

	// Filter.
	const size_t order = 4;  // Filter order.
	const double Fc1 = 10.0;  // Lower cutoff frequency [Hz].
	const double Fc2 = 40.0;  // Upper cutoff frequency [Hz].
	const double Fs = samplingRate;  // Sampling rate [Hz].

	const double f1f = 2.0 * Fc1 / Fs;  // Lower cutoff frequency as a fraction of Pi [0, 1].
	const double f2f = 2.0 * Fc2 / Fs;  // Upper cutoff frequency as a fraction of Pi [0, 1].
	const bool scaleFlag = true;  // Scale flag: true to scale, false to not scale ccof.

	// Calculate the d coefficients.
	const double *dcof = dcof_bwbp(order, f1f, f2f);  // d coefficients.
	if (nullptr == dcof)
	{
		std::cerr << "Unable to calculate d coefficients" << std::endl;
		return;
	}

	// Calculate the c coefficients.
	const int *ccof = ccof_bwbp(order);  // c coefficients.
	if (nullptr == ccof)
	{
		std::cerr << "Unable to calculate c coefficients" << std::endl;
		return;
	}

	// Output the c coefficients.
	std::cout << "Number of c coefficients (numerator) = " << 2 * order + 1 << std::endl;  // Number of c coefficients.
	std::vector<double> ccoeff;
	ccoeff.reserve(2 * order + 1);
	if (!scaleFlag)
	{
		for (size_t i = 0; i <= 2 * order; ++i)
		{
			ccoeff.push_back(ccof[i]);
			std::cout << ccof[i] << ", ";
		}
		std::cout << std::endl;
	}
	else
	{
		const double sf = sf_bwbp(order, f1f, f2f);  // Scaling factor for the c coefficients.

		for (size_t i = 0; i <= 2 * order; ++i)
		{
			ccoeff.push_back((double)ccof[i] * sf);
			std::cout << (double)ccof[i] * sf << ", ";
		}
		std::cout << std::endl;
	}

	// Output the d coefficients.
	std::cout << "Number of d coefficients (denominator) = " << 2 * order + 1 << std::endl;;  // Number of d coefficients.
	for (size_t i = 0; i <= 2 * order; ++i)
		std::cout << dcof[i] << ", ";
	std::cout << std::endl;

	// Filter signal.
	std::vector<double> filtered_signal;
	//my_common_api::filter(std::vector<double>(dcof, dcof + 2 * order + 1), std::vector<double>(ccof, ccof + 2 * order + 1), signal, filtered_signal);
	my_common_api::filter(std::vector<double>(dcof, dcof + 2 * order + 1), ccoeff, signal, filtered_signal);
	my_common_api::write_signal(filtered_signal, "./data/signal_processing/after_butterworth_bandpass_filtering2.txt");

	// Clean-up.
	free((void *)dcof);
	free((void *)ccof);
}

}  // namespace local
}  // unnamed namespace

namespace my_sigproc {

}  // namespace my_sigproc

int sigproc_main(int argc, char *argv[])
{
	// Example.
	local::butterworth_low_pass_filter_coefficients();
	local::butterworth_band_pass_filter_coefficients();

	return 0;
}
