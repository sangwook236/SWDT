//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_hmm {

void hmm_sample();
void hmm_forward_backward();
void hmm_viterbi();
void hmm_learning();

}  // namespace my_hmm

int hmm_main(int argc, char *argv[])
{
	//my_hmm::hmm_sample();
	//my_hmm::hmm_forward_backward();
	//my_hmm::hmm_viterbi();
	my_hmm::hmm_learning();

	return 0;
}
