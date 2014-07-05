//#include "stdafx.h"
#include <spuc/max_pn.h>
#include <spuc/delay.h>
#include <spuc/fir.h>
#include <spuc/mle.h>
#include <fstream>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

// [ref] ${SPUC_HOME}/examples/test_mlse.cpp
void maximum_likelihood_sequence_estimation_example()
{
	const char taps = 3;

	std::ofstream resf("./data/signal_processing/spuc/mlse.dat");
	std::ofstream reff("./data/signal_processing/spuc/refmlse.dat");
	std::ofstream rawf("./data/signal_processing/spuc/raw.dat");

	SPUC::fir<SPUC::float_type, SPUC::float_type> tx_fir(taps);
	SPUC::delay<bool> delayed_ref(33);
	tx_fir.settap(0, 1.0);
	tx_fir.settap(1, -1.5);
	tx_fir.settap(2, 0.5);
	SPUC::mle<SPUC::float_type> viterbi(taps);
	for (int i = 0; i < taps; ++i)
	{
		viterbi.cfir.settap(i, tx_fir.coeff[i]);
	}

	SPUC::max_pn pn(0x006d, 63, -1);  // Maximal length PN sequence for data.
	for (int i = 0; i < 20; ++i) pn.out();

	long error = 0;
	for (int i = 0; i < 132; ++i)
	{
		const char data = pn.out();
		const bool ref_data = (data == 1) ? 1 : 0;
		const bool ref_dly = delayed_ref.update(ref_data);
		const SPUC::float_type tx_data = tx_fir.update(data);
		const SPUC::float_type rx_data = tx_data; // + noise?
		const long path_out = viterbi.mlsd(rx_data);
		const bool bit_out = (path_out & 0x80000000) ? 1 : 0;
		rawf << rx_data << std::endl;
		reff << ref_dly;
		resf << bit_out;
		error += (ref_dly != bit_out);
		std::cout << std::setw(8) << std::setfill('0') << std::hex << path_out << ' ' << error << std::endl;
	}

	reff.close();
	resf.close();
	rawf.close();
}

}  // namespace my_spuc
