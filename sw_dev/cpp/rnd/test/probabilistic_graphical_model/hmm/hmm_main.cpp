//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace hmm {

void hmm_sample();
void hmm_forward_backward();
void hmm_viterbi();
void hmm_learning();

}  // namespace hmm

int hmm_main(int argc, char *argv[])
{
	//hmm::hmm_sample();
	//hmm::hmm_forward_backward();
	//hmm::hmm_viterbi();
	hmm::hmm_learning();

	return 0;
}
