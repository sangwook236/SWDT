//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_caffe {

void classification_example();

}  // namespace my_caffe

int caffe_main(int argc, char *argv[])
{
    // Train.
    // REF [site] >> http://caffe.berkeleyvision.org/tutorial/solver.html
    //  use the executable, caffe.

    // Predict.
	my_caffe::classification_example();

	// Feature extraction.

	return 0;
}
