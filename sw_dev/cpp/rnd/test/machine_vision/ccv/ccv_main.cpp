//include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_ccv {

void sift();  // scale invariant feature transform (SIFT)
void mser();  // maximally stable extremal regions (MSER)
void hog();  // histogram of oriented gradients (HOG)
void daisy();
void ferns();
void swt();  // stroke width transform (SWT)
void bbf();  // brightness binary feature (BBF)

void dpm();  // deformable parts model (DPM)
void tld();  // track learn detect (TLD)

void sparse_coding();
void compressive_sensing();

}  // namespace my_ccv

int ccv_main(int argc, char *argv[])
{
	// feature analysis -----------------------------------
	{
		//my_ccv::sift();  // not yet implemented
		//my_ccv::mser();  // not yet implemented
		//my_ccv::hog();  // not yet implemented
		my_ccv::daisy();
		//my_ccv::ferns();  // not yet implemented
		//my_ccv::swt();  // not yet implemented
		//my_ccv::bbf();  // not yet implemented
	}

	// object detection & tracking-------------------------
	{
		//my_ccv::dpm();  // not yet implemented
		//my_ccv::tld();  // not yet implemented
	}

	// sparse coding & compressive sensing ----------------
	{
		//my_ccv::sparse_coding();  // not yet implemented
		//my_ccv::compressive_sensing();  // not yet implemented
	}

	return 0;
}
