//#include "stdafx.h"
#include "umdhmm_nrutil.h"
#include "umdhmm_hmm.h"
#include "umdhmm_cdhmm.h"
#include "viterbi.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdio>

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

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	viterbi.c & testvit.c
void hmm_with_discrete_multinomial_observations__viterbi_umdhmm()
{
	umdhmm::HMM hmm;
	{
#if 1
		hmm.N = 3;  // the number of hidden states
		hmm.M = 2;  // the number of observation symbols
		const double pi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double A[] = {
			0.9,  0.05, 0.05,
			0.45, 0.1,  0.45,
			0.45, 0.45, 0.1
		};
		const double B[] = {
			0.5,   0.5,
			0.75,  0.25,
			0.25,  0.75
		};
#else
		hmm.N = 3;  // the number of hidden states
		hmm.M = 2;  // the number of observation symbols
		const double pi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double A[] = {
			0.5, 0.2,  0.2,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
		const double B[] = {
			0.5,   0.5,
			0.75,  0.25,
			0.25,  0.75
		};
#endif

		hmm.pi = (double *)umdhmm::dvector(1, hmm.N);
		const double *ptr = pi;
		for (int i = 1; i <= hmm.N; ++i, ++ptr)
			hmm.pi[i] = *ptr;

		hmm.A = (double **)umdhmm::dmatrix(1, hmm.N, 1, hmm.N);
		ptr = A;
		for (int i = 1; i <= hmm.N; ++i)
		{
			for (int j = 1; j <= hmm.N; ++j, ++ptr)
				hmm.A[i][j] = *ptr;
		}

		hmm.B = (double **)umdhmm::dmatrix(1, hmm.N, 1, hmm.M);
		ptr = B;
		for (int j = 1; j <= hmm.N; ++j)
		{
			for (int k = 1; k <= hmm.M; ++k, ++ptr)
				hmm.B[j][k] = *ptr;
		}
	}

	//
#if 0
	const int T = 50;  // length of observation sequence, T
	int	*O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
	{
		// use 1-based index
		const int seq[] = {
			2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1,
		};

		for (int i = 1; i <= T; ++i)
			O[i] = seq[i - 1];
	}
#elif 0
	const int T = 100;  // length of observation sequence, T
	int	*O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
	{
		// use 1-based index
		const int seq[] = {
			2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 
		};

		for (int i = 1; i <= T; ++i)
			O[i] = seq[i - 1];
	}
#elif 1
	const int T = 1500;  // length of observation sequence, T
	int	*O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
	{
		// use 1-based index
		const int seq[] = {
			2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 
		};

		for (int i = 1; i <= T; ++i)
			O[i] = seq[i - 1];
	}
#endif

	//
	int	*q = umdhmm::ivector(1, T);

	double **delta = umdhmm::dmatrix(1, T, 1, hmm.N);
	int	**psi = umdhmm::imatrix(1, T, 1, hmm.N);

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Viterbi using direct probabilities" << std::endl;

	double proba = 0.0; 
	umdhmm::Viterbi(&hmm, T, O, delta, psi, q, &proba);

	std::cout << "Viterbi MLE log prob = " << std::scientific << std::log(proba) << std::endl;
	std::cout << "Optimal state sequence:" << std::endl;
	umdhmm::PrintSequence(stdout, T, q);

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Viterbi using log probabilities" << std::endl;

	// note: ViterbiLog() returns back with log(A[i][j]) instead of leaving the A matrix alone.
	//	If you need the original A, you can make a copy of hmm by calling CopyHMM

	double logproba = 0.0; 
	umdhmm::ViterbiLog(&hmm, T, O, delta, psi, q, &logproba); 

	std::cout << "Viterbi MLE log prob = " << std::scientific << logproba << std::endl;
	std::cout << "Optimal state sequence:" << std::endl;
	umdhmm::PrintSequence(stdout, T, q);

	std::cout << "------------------------------------" << std::endl;
	std::cout << "The two log probabilites and optimal state sequences" << std::endl;
	std::cout << "should identical (within numerical precision)." << std::endl;

	//
	umdhmm::free_ivector(q, 1, T);
	umdhmm::free_ivector(O, 1, T);
	umdhmm::free_imatrix(psi, 1, T, 1, hmm.N);
	umdhmm::free_dmatrix(delta, 1, T, 1, hmm.N);
	umdhmm::FreeHMM(&hmm);
}

void cdhmm_with_gaussian_observations__viterbi_umdhmm()
{
	umdhmm::HMM hmm;
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_gaussian_mixture_observations__viterbi_umdhmm()
{
	umdhmm::HMM hmm;
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void hmm_viterbi()
{
	//local::viterbi_algorithm_1();
	//local::viterbi_algorithm_2();

	local::hmm_with_discrete_multinomial_observations__viterbi_umdhmm();
	//local::cdhmm_with_gaussian_observations__viterbi_umdhmm();  // not yet implemented
	//local::cdhmm_with_gaussian_mixture_observations__viterbi_umdhmm();  // not yet implemented
}
