//#include "stdafx.h"
#include "DspFilters/Filter.h"
#include "DspFilters/Butterworth.h"
#include "DspFilters/ChebyshevI.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <cmath>
#if defined(_WIN64) || defined(_WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif


namespace {
namespace local {

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

void simple_example()
{
    const int MaxOrder = 8;
    const int Channels = 1;

	// Signal.
	const double samplingRate = 1000.0;  // [Hz].
	const double startTime = 0.0;  // [sec].
	const double endTime = 2.0;  // [sec].

	// Filter.
	const int order = 4;
	const double LP = 10.0;  // Low cutoff frequency.
	const double HP = 40.0;  // High cutoff frequency.

	// REF [site] >> https://github.com/vinniefalco/DSPFilters
	{
		std::vector<double> signal;
		generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
		write_signal(signal, "./data/signal_processing/before_chebyshev_filtering.txt");

		// Create a Chebyshev type I bandstop filter of order 'order' with state for processing 'channels' channels of audio.
		Dsp::SimpleFilter<Dsp::ChebyshevI::BandStop<MaxOrder>, Channels> filter;
		filter.setup(order,  // Order.
			samplingRate,  // Sampling rate.
			(LP + HP) * 0.5,  // Center frequency.
			HP - LP,  // Band width.
			1);  // Ripple dB.

		// Filter signal.
		double *signals[] = { &signal[0], };
		filter.process((int)signal.size(), signals);

		write_signal(signal, "./data/signal_processing/after_chebyshev_filtering.txt");

/*
		% Visualize in Matlab.
		fid = fopen('before_chebyshev_filtering.txt', 'r');
		[y1, count1] = fscanf(fid, ['%lf']);
		fclose(fid);
		fid = fopen('after_chebyshev_filtering.txt', 'r');
		[y2, count2] = fscanf(fid, ['%lf']);
		fclose(fid);

		figure; plot(1:count1,y1)
		figure; plot(1:count2,y2)
*/
	}

	{
		std::vector<double> signal;
		generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
		write_signal(signal, "./data/signal_processing/before_butterworth_filtering.txt");

		// Create a Butterworth bandpass filter of order 'order' with state for processing 'channels' channels of audio.
		Dsp::SimpleFilter<Dsp::Butterworth::BandPass<MaxOrder>, Channels> filter;
		filter.setup(order,  // Order.
			samplingRate,  // Sampling rate.
			(LP + HP) * 0.5,  // Center frequency.
			HP - LP);  // Band width.

		// Filter signal.
		double *signals[] = { &signal[0], };
		filter.process((int)signal.size(), signals);

		write_signal(signal, "./data/signal_processing/after_butterworth_filtering.txt");

/*
		% Visualize in Matlab.
		fid = fopen('before_butterworth_filtering.txt', 'r');
		[y1, count1] = fscanf(fid, ['%lf']);
		fclose(fid);
		fid = fopen('after_butterworth_filtering.txt', 'r');
		[y2, count2] = fscanf(fid, ['%lf']);
		fclose(fid);

		figure; plot(1:count1,y1)
		figure; plot(1:count2,y2)
*/
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_dspfilters {

}  // namespace my_dspfilters

int dspfilters_main(int argc, char *argv[])
{
	// Example.
	local::simple_example();

	return 0;
}
