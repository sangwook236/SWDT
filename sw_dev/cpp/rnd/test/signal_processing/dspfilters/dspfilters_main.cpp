//#include "stdafx.h"
#include "DspFilters/Filter.h"
#include "DspFilters/Butterworth.h"
#include "DspFilters/ChebyshevI.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>


namespace my_common_api {

void generate_signal(const double startTime, const double endTime, const double interval, std::vector<double> &signal);
void write_signal(const std::vector<double> &signal, const std::string &filepath);

}  // namespace my_common_api

namespace {
namespace local {

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
		my_common_api::generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
		my_common_api::write_signal(signal, "./data/signal_processing/before_chebyshev_bandstop_filtering.txt");

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

		my_common_api::write_signal(signal, "./data/signal_processing/after_chebyshev_bandstop_filtering.txt");

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
		my_common_api::generate_signal(startTime, endTime, 1.0 / samplingRate, signal);
		my_common_api::write_signal(signal, "./data/signal_processing/before_butterworth_bandpass_filtering.txt");

		// Create a Butterworth bandpass filter of order 'order' with state for processing 'channels' channels of audio.
		Dsp::SimpleFilter<Dsp::Butterworth::BandPass<MaxOrder>, Channels> filter;
		filter.setup(order,  // Order.
			samplingRate,  // Sampling rate.
			(LP + HP) * 0.5,  // Center frequency.
			HP - LP);  // Band width.

		// Filter signal.
		double *signals[] = { &signal[0], };
		filter.process((int)signal.size(), signals);

		my_common_api::write_signal(signal, "./data/signal_processing/after_butterworth_bandpass_filtering.txt");

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
