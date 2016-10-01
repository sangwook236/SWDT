//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ccv {

void sift();  // Scale invariant feature transform (SIFT).
void mser();  // Maximally stable extremal regions (MSER).
void hog();  // Histogram of oriented gradients (HOG).
void daisy();
void ferns();
void swt();  // Stroke width transform (SWT).
void bbf();  // Brightness binary feature (BBF).
void icf();  // Integral channel features (ICF).

void dpm();  // Deformable parts model (DPM).
void tld();  // Track learn detect (TLD).

void sparse_coding();
void compressive_sensing();

}  // namespace my_ccv

int ccv_main(int argc, char *argv[])
{
	// Feature analysis ---------------------------------------------
	{
		//my_ccv::sift();  // Not yet implemented.
		//my_ccv::mser();  // Not yet implemented.
		//my_ccv::hog();  // Not yet implemented.
		my_ccv::daisy();
		//my_ccv::ferns();  // Not yet implemented.
		//my_ccv::swt();  // Not yet implemented.
		//my_ccv::bbf();  // Not yet implemented.
		//my_ccv::icf();  // Not yet implemented.
	}

	// Object detection & tracking-----------------------------------
	{
		//my_ccv::dpm();  // Not yet implemented.
		//my_ccv::tld();  // Not yet implemented.
	}

	// Sparse coding & compressive sensing --------------------------
	{
		//my_ccv::sparse_coding();  // Not yet implemented.
		//my_ccv::compressive_sensing();  // Not yet implemented.
	}

	return 0;
}
