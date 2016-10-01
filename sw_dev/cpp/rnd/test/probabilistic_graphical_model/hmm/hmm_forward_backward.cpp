//#include "stdafx.h"
#include "../umdhmm_lib/umdhmm_nrutil.h"
#include "../umdhmm_lib/umdhmm_hmm.h"
#include "../umdhmm_lib/umdhmm_cdhmm.h"
#include <iostream>
#include <stdexcept>


//#define __TEST_HMM_MODEL 1
#define __TEST_HMM_MODEL 2


namespace {
namespace local {

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	forward.c & testfor.c
void hmm_with_discrete_multinomial_observations__forward_umdhmm()
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

	//
#if 1
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
#elif 0
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

	//
	double **alpha = umdhmm::dmatrix(1, T, 1, hmm.N);

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Forward without scaling" << std::endl;

	double proba = 0.0; 
	umdhmm::Forward(&hmm, T, O, alpha, &proba); 
	std::cout << "log prob(O | model) = " << std::scientific << std::log(proba) << std::endl;

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Forward with scaling" << std::endl;

	double *scale = umdhmm::dvector(1, T);
	double logproba = 0.0; 
	umdhmm::ForwardWithScale(&hmm, T, O, alpha, scale, &logproba); 
	std::cout << "log prob(O | model) = " << std::scientific << logproba << std::endl;

	std::cout << "------------------------------------" << std::endl;
	std::cout << "The two log probabilites should identical (within numerical precision)." << std::endl;
	std::cout << "When observation sequence is very large, use scaling." << std::endl;

	//
	umdhmm::free_ivector(O, 1, T);
	umdhmm::free_dmatrix(alpha, 1, T, 1, hmm.N);
	umdhmm::free_dvector(scale, 1, T);
	umdhmm::FreeHMM(&hmm);
}

// [ref] umdhmm (http://www.kanungo.com/software/software.html#umdhmm)
//	backward.c
void hmm_with_discrete_multinomial_observations__backward_umdhmm()
{
	throw std::runtime_error("Not yet implemented");
}

void cdhmm_with_univariate_gaussian_observations__forward_umdhmm()
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

	//
	int T = 0;
	int M = 0;
	double **O = NULL;
	{
#if __TEST_HMM_MODEL == 1
		FILE *fp = fopen("./data/probabilistic_graphical_model/t1_uni_normal_50.seq", "r");
		//FILE *fp = fopen("./data/probabilistic_graphical_model/t1_uni_normal_100.seq", "r");
		//FILE *fp = fopen("./data/probabilistic_graphical_model/t1_uni_normal_1500.seq", "r");
#elif __TEST_HMM_MODEL == 2
		FILE *fp = fopen("./data/probabilistic_graphical_model/t2_uni_normal_50.seq", "r");
		//FILE *fp = fopen("./data/probabilistic_graphical_model/t2_uni_normal_100.seq", "r");
		//FILE *fp = fopen("./data/probabilistic_graphical_model/t2_uni_normal_1500.seq", "r");
#endif
		umdhmm::ReadSequence(fp, &T, &M, &O);
		fclose(fp);
	}

	//
	double **alpha = umdhmm::dmatrix(1, T, 1, cdhmm.N);

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Forward without scaling" << std::endl;

	double proba = 0.0;
	umdhmm::Forward(&cdhmm, T, O, alpha, &proba); 
	std::cout << "log prob(O | model) = " << std::scientific << std::log(proba) << std::endl;

	//
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Forward with scaling" << std::endl;

	double *scale = umdhmm::dvector(1, T);
	double logproba = 0.0;
	umdhmm::ForwardWithScale(&cdhmm, T, O, alpha, scale, &logproba); 
	std::cout << "log prob(O | model) = " << std::scientific << logproba << std::endl;

	std::cout << "------------------------------------" << std::endl;
	std::cout << "The two log probabilites should identical (within numerical precision)." << std::endl;
	std::cout << "When observation sequence is very large, use scaling." << std::endl;

	//
	umdhmm::free_dmatrix(O, 1, T, 1, cdhmm.M);
	umdhmm::free_dmatrix(alpha, 1, T, 1, cdhmm.N);
	umdhmm::free_dvector(scale, 1, T);
	umdhmm::FreeCDHMM_UnivariateNormal(&cdhmm);
}

void cdhmm_with_univariate_gaussian_observations__backward_umdhmm()
{
	throw std::runtime_error("Not yet implemented");
}

void cdhmm_with_univariate_gaussian_mixture_observations__forward_umdhmm()
{
	throw std::runtime_error("Not yet implemented");
}

void cdhmm_with_univariate_gaussian_mixture_observations__backward_umdhmm()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_hmm {

void hmm_forward_backward()
{
    local::hmm_with_discrete_multinomial_observations__forward_umdhmm();
    //local::hmm_with_discrete_multinomial_observations__backward_umdhmm();  // Not yet implemented.

	local::cdhmm_with_univariate_gaussian_observations__forward_umdhmm();
    //local::cdhmm_with_univariate_gaussian_observations__backward_umdhmm();  // Not yet implemented.

	//local::cdhmm_with_univariate_gaussian_mixture_observations__forward_umdhmm();  // Not yet implemented.
    //local::cdhmm_with_univariate_gaussian_mixture_observations__backward_umdhmm();  // Not yet implemented.
}

}  // namespace my_hmm
