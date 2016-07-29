//#include "stdafx.h"
#include <spuc/max_pn.h>
#include <spuc/noise.h>
#include <spuc/delay.h>
#include <spuce/filters/fir.h>
#include <spuc/mle.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

// REF [file] >> ${SPUC_HOME}/examples/test_mlse.cpp
void maximum_likelihood_sequence_estimation_example()
{
	const char taps = 3;

	std::ofstream resf("./data/signal_processing/spuc/mlse.dat");
	std::ofstream reff("./data/signal_processing/spuc/refmlse.dat");
	std::ofstream rawf("./data/signal_processing/spuc/raw.dat");

	spuce::fir<spuce::float_type, spuce::float_type> tx_fir(taps);
	// Necessary to account for delay through MLSE.
	SPUC::delay<bool> delayed_ref(34);
	// "Typical" Channel impulse response.
	tx_fir.settap(0, 1.0);
	tx_fir.settap(1, -1.5);
	tx_fir.settap(2, 0.5);
	// The Equalizer.
	SPUC::mle<SPUC::float_type> viterbi(taps);
	// Pre-set Taps as would occur during 'training'.
	for (int i = 0; i < taps; ++i)
	{
		viterbi.cfir.settap(i, tx_fir.gettap(i));
	}
	// Randomized levels.
	SPUC::noise rando;
	SPUC::max_pn pn(0x006d, 63, -1);  // Maximal length PN sequence for data.
	for (int i = 0; i < 20; ++i) pn.out();

	const int n = 100;
	std::vector<double> x(n), y(n), z(n);

	// Start without noise.
	SPUC::float_type noise_gain = 0;
	long error = 0;
	int i = 1;
	while (true)
	{
		const char data = pn.out();
		const bool ref_data = (data == 1) ? 1 : 0;
		const bool ref_dly = delayed_ref.update(ref_data);
		const SPUC::float_type tx_data = tx_fir.update(data);
		const SPUC::float_type rx_data = tx_data + noise_gain * rando.gauss(); // + noise.
		const long path_out = viterbi.mlsd(rx_data);
		const bool bit_out = (path_out & 0x80000000) ? 1 : 0;
		error += (ref_dly != bit_out);
		// Fast scrolling but not too fast.
		std::chrono::milliseconds sec(10);
		std::this_thread::sleep_for(sec);
		i++;
		// Randomize noise gain.
		if (0 == i % 400)
		{
			noise_gain = 2.0 * rando.uniform();
			//error = 0;  // Reset error rate.
			std::cout << "Noise gain = " << noise_gain << " BER = " << error / (double)(i) << "\n";
		}
	}
}

}  // namespace my_spuc
