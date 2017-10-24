//#include "stdafx.h"
#include <aquila/aquila.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>


namespace {
namespace local {

// REF [file] >> ${AQUILA_HOME}/examples/fft_simple_spectrum/fft_simple_spectrum.cpp.
void fft_simple_spectrum_example()
{
	// Input signal parameters.
	const std::size_t SIZE = 64;
	const Aquila::FrequencyType sampleFreq = 2000, f1 = 125, f2 = 700;

	Aquila::SineGenerator sine1(sampleFreq);
	sine1.setAmplitude(32).setFrequency(f1).generate(SIZE);
	Aquila::SineGenerator sine2(sampleFreq);
	sine2.setAmplitude(8).setFrequency(f2).setPhase(0.75).generate(SIZE);
	auto sum = sine1 + sine2;

	Aquila::TextPlot plot("Input signal");
	plot.plot(sum);

	// Calculate the FFT.
	auto fft = Aquila::FftFactory::getFft(SIZE);
	Aquila::SpectrumType spectrum = fft->fft(sum.toArray());

	plot.setTitle("Spectrum");
	plot.plotSpectrum(spectrum);
}

// REF [file] >> ${AQUILA_HOME}/examples/fft_filter/fft_filter.cpp.
void fft_filter_example()
{
	// Input signal parameters.
	const std::size_t SIZE = 64;
	const Aquila::FrequencyType sampleFreq = 2000;
	const Aquila::FrequencyType f1 = 96, f2 = 813;
	const Aquila::FrequencyType f_lp = 500;

	Aquila::SineGenerator sineGenerator1(sampleFreq);
	sineGenerator1.setAmplitude(32).setFrequency(f1).generate(SIZE);
	Aquila::SineGenerator sineGenerator2(sampleFreq);
	sineGenerator2.setAmplitude(8).setFrequency(f2).setPhase(0.75).generate(SIZE);
	auto sum = sineGenerator1 + sineGenerator2;

	Aquila::TextPlot plt("Signal waveform before filtration");
	plt.plot(sum);

	// Calculate the FFT.
	auto fft = Aquila::FftFactory::getFft(SIZE);
	Aquila::SpectrumType spectrum = fft->fft(sum.toArray());
	plt.setTitle("Signal spectrum before filtration");
	plt.plotSpectrum(spectrum);

	// Generate a low-pass filter spectrum.
	Aquila::SpectrumType filterSpectrum(SIZE);
	for (std::size_t i = 0; i < SIZE; ++i)
	{
		if (i < (SIZE * f_lp / sampleFreq))
		{
			// Passband.
			filterSpectrum[i] = 1.0;
		}
		else
		{
			// Stopband.
			filterSpectrum[i] = 0.0;
		}
	}
	plt.setTitle("Filter spectrum");
	plt.plotSpectrum(filterSpectrum);

	// The following call does the multiplication of two spectra (which is complementary to convolution in time domain).
	std::transform(
		std::begin(spectrum),
		std::end(spectrum),
		std::begin(filterSpectrum),
		std::begin(spectrum),
		[](Aquila::ComplexType x, Aquila::ComplexType y) { return x * y; }
	);
	plt.setTitle("Signal spectrum after filtration");
	plt.plotSpectrum(spectrum);

	// Inverse FFT moves us back to time domain.
	double x1[SIZE];
	fft->ifft(spectrum, x1);
	plt.setTitle("Signal waveform after filtration");
	plt.plot(x1, SIZE);
}

// REF [file] >> ${AQUILA_HOME}/examples/mfcc_calculation/mfcc_calculation.cpp.
void mfcc_calculation_example()
{
	const std::size_t SIZE = 1024;
	const Aquila::FrequencyType sampleFrequency = 1024;

	Aquila::SineGenerator input(sampleFrequency);
	input.setAmplitude(5).setFrequency(64).generate(SIZE);

	Aquila::Mfcc mfcc(input.getSamplesCount());
	const std::size_t numFeatures = 12;
	const auto &mfccValues = mfcc.calculate(input, numFeatures);
	std::cout << "Mel-frequency cepstrum coefficients (MFCCs):" << std::endl;
	std::copy(std::begin(mfccValues), std::end(mfccValues), std::ostream_iterator<double>(std::cout, " "));
	std::cout << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_aquila {

}  // namespace my_aquila

int aquila_main(int argc, char *argv[])
{
	// Example.
	//local::fft_simple_spectrum_example();
	//local::fft_filter_example();

	// Mel-frequency cepstral coefficient (MFCC).
	local::mfcc_calculation_example();

	return 0;
}
