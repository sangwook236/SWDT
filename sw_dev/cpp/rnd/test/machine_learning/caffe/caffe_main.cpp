#define CPU_ONLY 1

//#include "stdafx.h"
#include <iostream>
#include <caffe/caffe.hpp>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_caffe {

void classification_example();

}  // namespace my_caffe

// REF [site] >>
//	http://caffe.berkeleyvision.org/tutorial/
//	http://caffe.berkeleyvision.org/tutorial/layers.html
//	http://caffe.berkeleyvision.org/tutorial/loss.html
//	http://caffe.berkeleyvision.org/tutorial/solver.html
//
//	https://github.com/BVLC/caffe/wiki/Training-and-Resuming
//	https://github.com/BVLC/caffe/wiki/Faster-Caffe-Training

int caffe_main(int argc, char *argv[])
{
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	// Train.
	//  Use the executable, caffe.

	// Predict.
	//my_caffe::classification_example();

	// Feature extraction.

	return 0;
}
