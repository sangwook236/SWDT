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
//	baum.c & esthmm.c
void hmm_with_discrete_multinomial_observations__em_for_mle_umdhmm()
{
	umdhmm::HMM hmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed. 
*/

	// initialize the HMM model
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{ 
#if __TEST_HMM_MODEL == 1
		hmm.N = 3;  // the number of hidden states
		hmm.M = 2;  // the number of observation symbols
		const double initPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double initA[] = {
			0.9,  0.05, 0.05,
			0.45, 0.1,  0.45,
			0.45, 0.45, 0.1
		};
		const double initB[] = {
			0.5,   0.5,
			0.75,  0.25,
			0.25,  0.75
		};
#elif __TEST_HMM_MODEL == 2
		hmm.N = 3;  // the number of hidden states
		hmm.M = 2;  // the number of observation symbols
		const double initPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double initA[] = {
			0.5, 0.2,  0.3,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
		const double initB[] = {
			0.5,   0.5,
			0.75,  0.25,
			0.25,  0.75
		};
#endif

		hmm.pi = (double *)umdhmm::dvector(1, hmm.N);
		const double *ptr = initPi;
		for (int i = 1; i <= hmm.N; ++i, ++ptr)
			hmm.pi[i] = *ptr;

		hmm.A = (double **)umdhmm::dmatrix(1, hmm.N, 1, hmm.N);
		ptr = initA;
		for (int i = 1; i <= hmm.N; ++i)
		{
			for (int j = 1; j <= hmm.N; ++j, ++ptr)
				hmm.A[i][j] = *ptr;
		}

		hmm.B = (double **)umdhmm::dmatrix(1, hmm.N, 1, hmm.M);
		ptr = initB;
		for (int j = 1; j <= hmm.N; ++j)
		{
			for (int k = 1; k <= hmm.M; ++k, ++ptr)
				hmm.B[j][k] = *ptr;
		}
	}
	else if (2 == initialization_mode)
	{
		const int N = 3;  // the number of hidden states
		const int M = 2;  // the number of observation symbols
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const int seed = 34586u;
#else
		const int seed = umdhmm::hmmgetseed();
#endif
		umdhmm::InitHMM(&hmm, N, M, seed);

		std::cout << "random seed = " << seed << std::endl;
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	//
#if 0
	const int T = 50;  // length of observation sequence, T
	int *O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
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
	int *O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
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
	int *O = umdhmm::ivector(1, T);  // observation sequence O[1..T]
	{
		// use 1-based index
		const int seq[] = {
			2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 
		};

		for (int i = 1; i <= T; ++i)
			O[i] = seq[i - 1];
	}
#endif

	// allocate memory
	double **alpha = umdhmm::dmatrix(1, T, 1, hmm.N);
	double **beta = umdhmm::dmatrix(1, T, 1, hmm.N);
	double **gamma = umdhmm::dmatrix(1, T, 1, hmm.N);

	// call Baum-Welch
	int	numIterations = 0;
	double logProbInit = 0.0, logProbFinal = 0.0;
	const double terminationTolerance = 0.001;
	umdhmm::BaumWelch(&hmm, T, O, terminationTolerance, alpha, beta, gamma, &numIterations, &logProbInit, &logProbFinal);

	// compute gamma & xi
	{
		// gamma can use the result from umdhmm::BaumWelch
		//umdhmm::ComputeGamma(&hmm, T, alpha, beta, gamma);

		//
		double ***xi = umdhmm::AllocXi(T, hmm.N);
		umdhmm::ComputeXi(&hmm, T, O, alpha, beta, xi);
		umdhmm::FreeXi(xi, T, hmm.N);
	}

	// print the output 
	std::cout << "number of iterations = " << numIterations << std::endl;
	std::cout << "log prob(observation | init model) = " << std::scientific << logProbInit << std::endl;	
	std::cout << "log prob(observation | estimated model) = " << std::scientific << logProbFinal << std::endl;	

	umdhmm::PrintHMM(stdout, &hmm);

	// free memory
	umdhmm::free_ivector(O, 1, T);
	umdhmm::free_dmatrix(alpha, 1, T, 1, hmm.N);
	umdhmm::free_dmatrix(beta, 1, T, 1, hmm.N);
	umdhmm::free_dmatrix(gamma, 1, T, 1, hmm.N);
	umdhmm::FreeHMM(&hmm);
}

void hmm_with_discrete_multinomial_observations__em_for_map()
{
	throw std::runtime_error("not yet implemented");
}

void hmm_with_discrete_multinomial_observations__em_for_map_using_sparse_learning()
{
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_univariate_gaussian_observations__em_for_mle_umdhmm()
{
	umdhmm::CDHMM cdhmm;

/*
	you can initialize the hmm model three ways:
		1) with a model, which also sets the number of states N and number of symbols M.
		2) with a random model by just specifyin N and M.
		3) with a specific random model by specifying N, M and seed. 
*/

	// initialize the HMM model
	const int initialization_mode = 1;
	if (1 == initialization_mode)
	{ 
#if __TEST_HMM_MODEL == 1
		cdhmm.N = 3;  // the number of hidden states
		cdhmm.M = 1;  // the number of observation symbols
		const double initPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double initA[] = {
			0.9,  0.05, 0.05,
			0.45, 0.1,  0.45,
			0.45, 0.45, 0.1
		};
#elif __TEST_HMM_MODEL == 2
		cdhmm.N = 3;  // the number of hidden states
		cdhmm.M = 1;  // the number of observation symbols
		const double initPi[] = {
			1.0/3.0, 1.0/3.0, 1.0/3.0
		};
		const double initA[] = {
			0.5, 0.2,  0.3,
			0.2, 0.4,  0.4,
			0.1, 0.45, 0.45
		};
#endif

		cdhmm.pi = (double *)umdhmm::dvector(1, cdhmm.N);
		const double *ptr = initPi;
		for (int i = 1; i <= cdhmm.N; ++i, ++ptr)
			cdhmm.pi[i] = *ptr;

		cdhmm.A = (double **)umdhmm::dmatrix(1, cdhmm.N, 1, cdhmm.N);
		ptr = initA;
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
	else if (2 == initialization_mode)
	{
		const int N = 3;  // the number of hidden states
		const int M = 1;  // the number of observation symbols
#if defined(__USE_SPECIFIED_VALUE_FOR_RANDOM_SEED)
		const int seed = 34586u;
#else
		const int seed = umdhmm::hmmgetseed();
#endif
		umdhmm::InitCDHMM_UnivariateNormal(&cdhmm, N, M, seed);

		std::cout << "random seed = " << seed << std::endl;
	}
	else
		throw std::runtime_error("incorrect initialization mode");

	//
	int T = 0;
	int M = 0;
	double **O = NULL;
	{
#if __TEST_HMM_MODEL == 1
		//FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_50.seq", "r");
		//FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_100.seq", "r");
		FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t1_uni_normal_1500.seq", "r");
#elif __TEST_HMM_MODEL == 2
		//FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_50.seq", "r");
		//FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_100.seq", "r");
		FILE *fp = fopen(".\\probabilistic_graphical_model_data\\t2_uni_normal_1500.seq", "r");
#endif
		umdhmm::ReadSequence(fp, &T, &M, &O);
		fclose(fp);
	}

	// allocate memory
	double **alpha = umdhmm::dmatrix(1, T, 1, cdhmm.N);
	double **beta = umdhmm::dmatrix(1, T, 1, cdhmm.N);
	double **gamma = umdhmm::dmatrix(1, T, 1, cdhmm.N);

	// call Baum-Welch
	int	numIterations = 0;
	double logProbInit = 0.0, logProbFinal = 0.0;
	const double terminationTolerance = 0.001;
	umdhmm::BaumWelch_UnivariateNormal(&cdhmm, T, O, terminationTolerance, alpha, beta, gamma, &numIterations, &logProbInit, &logProbFinal);

	// compute gamma & xi
	{
		// gamma can use the result from umdhmm::BaumWelch
		//umdhmm::ComputeGamma(&hmm, T, alpha, beta, gamma);

		//
		double ***xi = umdhmm::AllocXi(T, cdhmm.N);
		umdhmm::ComputeXi(&cdhmm, T, O, alpha, beta, xi);
		umdhmm::FreeXi(xi, T, cdhmm.N);
	}

	// print the output 
	std::cout << "number of iterations = " << numIterations << std::endl;
	std::cout << "log prob(observation | init model) = " << std::scientific << logProbInit << std::endl;	
	std::cout << "log prob(observation | estimated model) = " << std::scientific << logProbFinal << std::endl;	

	umdhmm::PrintCDHMM_UnivariateNormal(stdout, &cdhmm);

	// free memory
	umdhmm::free_dmatrix(O, 1, T, 1, cdhmm.M);
	umdhmm::free_dmatrix(alpha, 1, T, 1, cdhmm.N);
	umdhmm::free_dmatrix(beta, 1, T, 1, cdhmm.N);
	umdhmm::free_dmatrix(gamma, 1, T, 1, cdhmm.N);
	umdhmm::FreeCDHMM_UnivariateNormal(&cdhmm);
}

void cdhmm_with_univariate_gaussian_observations__em_for_map()
{
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_univariate_gaussian_observations__em_for_map_using_sparse_learning()
{
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_univariate_gaussian_mixture_observations__em_for_mle_umdhmm()
{
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_univariate_gaussian_mixture_observations__em_for_map()
{
	throw std::runtime_error("not yet implemented");
}

void cdhmm_with_univariate_gaussian_mixture_observations__em_for_map_using_sparse_learning()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace hmm {

void hmm_learning()
{
    local::hmm_with_discrete_multinomial_observations__em_for_mle_umdhmm();
    //local::hmm_with_discrete_multinomial_observations__em_for_map();  // not yet implemented
    //local::hmm_with_discrete_multinomial_observations__em_for_map_using_sparse_learning();  // not yet implemented

    local::cdhmm_with_univariate_gaussian_observations__em_for_mle_umdhmm();  // not yet implemented
    //local::cdhmm_with_univariate_gaussian_observations__em_for_map();  // not yet implemented
    //local::cdhmm_with_univariate_gaussian_observations__em_for_map_using_sparse_learning();  // not yet implemented

    //local::cdhmm_with_univariate_gaussian_mixture_observations__em_for_mle_umdhmm();  // not yet implemented
    //local::cdhmm_with_univariate_gaussian_mixture_observations__em_for_map();  // not yet implemented
    //local::cdhmm_with_univariate_gaussian_mixture_observations__em_for_map_using_sparse_learning();  // not yet implemented
}

}  // namespace hmm
