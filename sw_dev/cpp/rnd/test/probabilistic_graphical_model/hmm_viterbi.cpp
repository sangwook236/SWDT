//#include "stdafx.h"
#include "viterbi.hpp"
#include <iostream>


void viterbi_algorithm();

namespace {
namespace local {

void viterbi_algorithm_1()
{
	//
	std::cout << "********** method 1" << std::endl;
	viterbi_algorithm();
}

void viterbi_algorithm_2()
{
	//
	std::cout << "\n********** method 2" << std::endl;
	Viterbi::HMM hmmObj;
	hmmObj.init();
	std::cout << hmmObj;

	Viterbi::forward_viterbi(hmmObj.get_observations(), hmmObj.get_states(), hmmObj.get_start_probability(), hmmObj.get_transition_probability(), hmmObj.get_emission_probability());
}

}  // namespace local
}  // unnamed namespace

void hmm_viterbi()
{
	local::viterbi_algorithm_1();
	local::viterbi_algorithm_2();
}

