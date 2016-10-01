/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   hmmutils.c
**      Purpose: utilities for reading, writing CDHMM stuff.
**      Organization: University of Maryland
**
**      $Id: hmmutils.c,v 1.4 1998/02/23 07:51:26 kanungo Exp kanungo $
*/

#include "umdhmm_nrutil.h"
#include "umdhmm_cdhmm.h"
#include "umdhmm_hmm.h"
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <cstdio>
#include <cstdlib>
#include <cmath>


namespace umdhmm {

static char rcsid[] = "$Id: hmmutils.c,v 1.4 1998/02/23 07:51:26 kanungo Exp kanungo $";

void ReadCDHMM_UnivariateNormal(FILE *fp, CDHMM *phmm)
{
	int i, j;

	fscanf(fp, "M= %d\n", &(phmm->M));  // the number of observation symbols
	fscanf(fp, "N= %d\n", &(phmm->N));  // the number of hidden states

	if (1 != phmm->M)
		throw std::runtime_error("the dimension of observation symbols is incorrect");

	fscanf(fp, "pi:\n");
	phmm->pi = (double *)dvector(1, phmm->N);
	for (i = 1; i <= phmm->N; ++i)
		fscanf(fp, "%lf", &(phmm->pi[i]));

	fscanf(fp, "A:\n");
	phmm->A = (double **)dmatrix(1, phmm->N, 1, phmm->N);
	for (i = 1; i <= phmm->N; ++i)
	{
		for (j = 1; j <= phmm->N; ++j)
			fscanf(fp, "%lf", &(phmm->A[i][j]));
		fscanf(fp,"\n");
	}

	fscanf(fp, "univariate normal:\n");
	UnivariateNormalParams *set_of_params = AllocSetOfParams_UnivariateNormal(1, phmm->N);
	for (j = 1; j <= phmm->N; ++j)
	{
		fscanf(fp, "%lf", &set_of_params[j].mean);
		fscanf(fp, "%lf", &set_of_params[j].stddev);
		fscanf(fp,"\n");
	}
	phmm->set_of_params = (void *)set_of_params;

	phmm->pdf = &umdhmm::univariate_normal_distribution;
}

void InitCDHMM_UnivariateNormal(CDHMM *phmm, int N, int M, int seed)
{
	if (1 != M)
		throw std::runtime_error("the dimension of observation symbols is incorrect");

	int i, j;
	double sum;

	// initialize random number generator

	hmmsetseed(seed);

	phmm->M = M;  // the number of observation symbols
	phmm->N = N;  // the number of hidden states

	phmm->pi = (double *)dvector(1, phmm->N);
	sum = 0.0;
	for (i = 1; i <= phmm->N; ++i)
	{
		phmm->pi[i] = hmmgetrand();
		sum += phmm->pi[i];
	}
	for (i = 1; i <= phmm->N; ++i)
		phmm->pi[i] /= sum;

	phmm->A = (double **)dmatrix(1, phmm->N, 1, phmm->N);
	for (i = 1; i <= phmm->N; ++i)
	{
		sum = 0.0;
		for (j = 1; j <= phmm->N; ++j)
		{
			phmm->A[i][j] = hmmgetrand();
			sum += phmm->A[i][j];
		}
		for (j = 1; j <= phmm->N; ++j)
			phmm->A[i][j] /= sum;
	}

	UnivariateNormalParams *set_of_params = AllocSetOfParams_UnivariateNormal(1, phmm->N);
	for (j = 1; j <= phmm->N; ++j)
	{
		set_of_params[j].mean = hmmgetrand(-10000.0, 10000.0);
		set_of_params[j].stddev = hmmgetrand(-10000.0, 10000.0);
	}
	phmm->set_of_params = (void *)set_of_params;

	phmm->pdf = &umdhmm::univariate_normal_distribution;
}

void CopyCDHMM_UnivariateNormal(CDHMM *phmm1, CDHMM *phmm2)
{
	if (1 != phmm1->M)
		throw std::runtime_error("the dimension of observation symbols is incorrect");

	int i, j;

	phmm2->M = phmm1->M;  // the number of observation symbols
	phmm2->N = phmm1->N;  // the number of hidden states

	phmm2->pi = (double *)dvector(1, phmm2->N);
	for (i = 1; i <= phmm2->N; ++i)
		phmm2->pi[i] = phmm1->pi[i];

	phmm2->A = (double **)dmatrix(1, phmm2->N, 1, phmm2->N);
	for (i = 1; i <= phmm2->N; ++i)
		for (j = 1; j <= phmm2->N; ++j)
			phmm2->A[i][j] = phmm1->A[i][j];

	UnivariateNormalParams *set_of_params1 = reinterpret_cast<UnivariateNormalParams *>(phmm1->set_of_params);
	UnivariateNormalParams *set_of_params2 = AllocSetOfParams_UnivariateNormal(1, phmm2->N);
	for (j = 1; j <= phmm2->N; ++j)
	{
		set_of_params2[j].mean = set_of_params1[j].mean;
		set_of_params2[j].stddev = set_of_params1[j].stddev;
	}
	phmm2->set_of_params = (void *)set_of_params2;

	phmm2->pdf = phmm1->pdf;
}

void PrintCDHMM_UnivariateNormal(FILE *fp, CDHMM *phmm)
{
	int i, j;

	fprintf(fp, "M= %d\n", phmm->M);
	fprintf(fp, "N= %d\n", phmm->N);

	fprintf(fp, "pi:\n");
	for (i = 1; i <= phmm->N; ++i)
		fprintf(fp, "%f ", phmm->pi[i]);
	fprintf(fp, "\n");

	fprintf(fp, "A:\n");
	for (i = 1; i <= phmm->N; ++i)
	{
		for (j = 1; j <= phmm->N; ++j)
			fprintf(fp, "%f ", phmm->A[i][j]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "univariate normal:\n");
	UnivariateNormalParams *set_of_params = reinterpret_cast<UnivariateNormalParams *>(phmm->set_of_params);
	for (j = 1; j <= phmm->N; ++j)
		fprintf(fp, "%lf %lf\n", set_of_params[j].mean, set_of_params[j].stddev);

	//phmm->pdf;
}

void FreeCDHMM_UnivariateNormal(CDHMM *phmm)
{
	free_dvector(phmm->pi, 1, phmm->N);
	phmm->pi = NULL;
	free_dmatrix(phmm->A, 1, phmm->N, 1, phmm->N);
	phmm->A = NULL;
	FreeSetOfParams_UnivariateNormal(phmm->set_of_params, 1, phmm->N);
	phmm->set_of_params = NULL;
	phmm->pdf = NULL;
}

double univariate_normal_distribution(const double *symbol, const int state, const void *set_of_parameters)
{
	const UnivariateNormalParams *set_of_params = reinterpret_cast<const UnivariateNormalParams *>(set_of_parameters);

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	boost::math::normal pdf(set_of_params[state].mean, set_of_params[state].stddev);

	return boost::math::pdf(pdf, symbol[1]);
}

double multivariate_normal_distribution(const double *symbol, const int state, const void *set_of_parameters)
{
	const MultivariateNormalParams *set_of_params = reinterpret_cast<const MultivariateNormalParams *>(set_of_parameters);

	//boost::math::normal pdf(set_of_params[state].mean, set_of_params[state].covar);

	throw std::runtime_error("Not yet implemented");
}

double von_mises_distribution(const double *symbol, const int state, const void *set_of_parameters)
{
	const vonMisesParams *set_of_params = reinterpret_cast<const vonMisesParams *>(set_of_parameters);

	throw std::runtime_error("Not yet implemented");
}

double von_mises_fisher_distribution(const double *symbol, const int state, const void *set_of_parameters)
{
	const vonMisesFisherParams *set_of_params = reinterpret_cast<const vonMisesFisherParams *>(set_of_parameters);

	throw std::runtime_error("Not yet implemented");
}

UnivariateNormalParams * AllocSetOfParams_UnivariateNormal(int nl, int nh)
{
	UnivariateNormalParams *s = (UnivariateNormalParams *)calloc(unsigned(nh - nl + 1), sizeof(UnivariateNormalParams));
	if (!s) nrerror("Allocation failure 1 in dmatrix()");
	s -= nl;

	return s;
}

void FreeSetOfParams_UnivariateNormal(void *s, int nl, int nh)
{
	free((char *)((UnivariateNormalParams *)s + nl));
}

}  // namespace umdhmm
