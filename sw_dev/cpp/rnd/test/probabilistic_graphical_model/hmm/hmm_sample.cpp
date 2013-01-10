//#include "stdafx.h"
#include "../umdhmm_lib/umdhmm_nrutil.h"
#include "../umdhmm_lib/umdhmm_hmm.h"
#include "../umdhmm_lib/umdhmm_cdhmm.h"
#include <iostream>
#include <stdexcept>


//#define __TEST_HMM_MODEL 1
#define __TEST_HMM_MODEL 2
#define __USE_SPECIFIED_VALUE_FOR_RANDOM_SEED 1


namespace {
namespace local {

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	genseq.c
void hmm_with_discrete_multinomial_observations__sample_umdhmm()
{
	umdhmm::HMM hmm;

	{
#if __TEST_HMM_MODEL == 1
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
#elif __TEST_HMM_MODEL == 2
		hmm.N = 3;  // the number of hidden states
		hmm.M = 2;  // the number of observation symbols
		const double pi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double A[] = {
			0.5, 0.2,  0.3,
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

#if 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_mutinomial_50.seq", "w");
	const int T = 50;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_mutinomial_100.seq", "w");
	const int T = 100;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_mutinomial_1500.seq", "w");
	const int T = 1500;  // length of observation sequence, T
#else
	FILE *fp = stdout;
	const int T = 100;  // length of observation sequence, T
#endif

	// set random number generator seed
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
	const int seed = 34586u;
#else
	const int seed = umdhmm::hmmgetseed();
#endif

	std::cout << "random seed = " << seed << std::endl;

	//
	int	*O = umdhmm::ivector(1, T);  // alloc space for observation sequence O[1..T]
	int	*q = umdhmm::ivector(1, T);  // alloc space for state sequence q[1..T]

	umdhmm::GenSequenceArray(&hmm, seed, T, O, q);

	//
	umdhmm::PrintSequence(stdout, T, O);

	//
	umdhmm::free_ivector(O, 1, T);
	umdhmm::free_ivector(q, 1, T);
	umdhmm::FreeHMM(&hmm);

	if (stdout != fp)
		fclose(fp);
}

void cdhmm_with_univariate_gaussian_observations__sample_umdhmm()
{
	umdhmm::CDHMM cdhmm;

	{
#if __TEST_HMM_MODEL == 1
		cdhmm.N = 3;  // the number of hidden states
		cdhmm.M = 1;  // the number of observation symbols
		const double pi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double A[] = {
			0.9,  0.05, 0.05,
			0.45, 0.1,  0.45,
			0.45, 0.45, 0.1
		};
#elif __TEST_HMM_MODEL == 2
		cdhmm.N = 3;  // the number of hidden states
		cdhmm.M = 1;  // the number of observation symbols
		const double pi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double A[] = {
			0.5, 0.2,  0.3,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
#endif

		cdhmm.pi = (double *)umdhmm::dvector(1, cdhmm.N);
		const double *ptr = pi;
		for (int i = 1; i <= cdhmm.N; ++i, ++ptr)
			cdhmm.pi[i] = *ptr;

		cdhmm.A = (double **)umdhmm::dmatrix(1, cdhmm.N, 1, cdhmm.N);
		ptr = A;
		for (int i = 1; i <= cdhmm.N; ++i)
		{
			for (int j = 1; j <= cdhmm.N; ++j, ++ptr)
				cdhmm.A[i][j] = *ptr;
		}

		{
			umdhmm::UnivariateNormalParams *set_of_params = umdhmm::AllocSetOfParams_UnivariateNormal(1, cdhmm.N);
#if __TEST_HMM_MODEL == 1
			set_of_params[1].mean = 0.0;
			set_of_params[1].stddev = 1.0;

			set_of_params[2].mean = 30.0;
			set_of_params[2].stddev = 2.0;

			set_of_params[3].mean = -20.0;
			set_of_params[3].stddev = 1.5;
#elif __TEST_HMM_MODEL == 2
			set_of_params[1].mean = 0.0;
			set_of_params[1].stddev = 1.0;

			set_of_params[2].mean = -30.0;
			set_of_params[2].stddev = 2.0;

			set_of_params[3].mean = 20.0;
			set_of_params[3].stddev = 1.5;
#endif
			cdhmm.set_of_params = (void *)set_of_params;
		}

		cdhmm.pdf = &umdhmm::univariate_normal_distribution;
	}

#if __TEST_HMM_MODEL == 1

#if 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_50.seq", "w");
	const int T = 50;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_100.seq", "w");
	const int T = 100;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_1500.seq", "w");
	const int T = 1500;  // length of observation sequence, T
#else
	FILE *fp = stdout;
	const int T = 100;  // length of observation sequence, T
#endif

#elif __TEST_HMM_MODEL == 2

#if 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_50.seq", "w");
	const int T = 50;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_100.seq", "w");
	const int T = 100;  // length of observation sequence, T
#elif 0
	FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_1500.seq", "w");
	const int T = 1500;  // length of observation sequence, T
#else
	FILE *fp = stdout;
	const int T = 100;  // length of observation sequence, T
#endif

#endif

	// set random number generator seed
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
	const int seed = 34586u;
#else
	const int seed = umdhmm::hmmgetseed();
#endif

	std::cout << "random seed = " << seed << std::endl;

	//
	double **O = umdhmm::dmatrix(1, T, 1, cdhmm.M);  // alloc space for observation sequence O[1..T]
	int *q = umdhmm::ivector(1, T);  // alloc space for state sequence q[1..T]

	umdhmm::GenSequenceArray_UnivariateNormal(&cdhmm, seed, T, O, q);

	//
	umdhmm::PrintSequence(fp, T, cdhmm.M, O);

	//
	umdhmm::free_dmatrix(O, 1, T, 1, cdhmm.M);
	umdhmm::free_ivector(q, 1, T);
	umdhmm::FreeCDHMM_UnivariateNormal(&cdhmm);

	if (stdout != fp)
		fclose(fp);
}

void cdhmm_with_univariate_gaussian_mixture_observations__sample_umdhmm()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_hmm {

void hmm_sample()
{
	//local::hmm_with_discrete_multinomial_observations__sample_umdhmm();
	local::cdhmm_with_univariate_gaussian_observations__sample_umdhmm();
	//local::cdhmm_with_univariate_gaussian_mixture_observations__sample_umdhmm();  // not yet implemented
}

}  // namespace my_hmm
