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
/*
void ReadHMM(FILE *fp, CDHMM *phmm)
{
	int i, j, k;

	fscanf(fp, "M= %d\n", &(phmm->M));
	fscanf(fp, "N= %d\n", &(phmm->N));

	fscanf(fp, "A:\n");
	phmm->A = (double **)dmatrix(1, phmm->N, 1, phmm->N);
	for (i = 1; i <= phmm->N; ++i)
	{
		for (j = 1; j <= phmm->N; ++j)
			fscanf(fp, "%lf", &(phmm->A[i][j]));
		fscanf(fp,"\n");
	}

	fscanf(fp, "B:\n");
	phmm->B = (double **)dmatrix(1, phmm->N, 1, phmm->M);
	for (j = 1; j <= phmm->N; ++j)
	{
		for (k = 1; k <= phmm->M; ++k)
			fscanf(fp, "%lf", &(phmm->B[j][k]));
		fscanf(fp,"\n");
	}

	fscanf(fp, "pi:\n");
	phmm->pi = (double *)dvector(1, phmm->N);
	for (i = 1; i <= phmm->N; ++i)
		fscanf(fp, "%lf", &(phmm->pi[i]));
}
*/

void FreeHMM(CDHMM *phmm)
{
	free_dvector(phmm->pi, 1, phmm->N);
	free_dmatrix(phmm->A, 1, phmm->N, 1, phmm->N);
	// FIXME [fix] >>
	//phmm->params =
	phmm->pdf = NULL;
	//free_dmatrix(phmm->B, 1, phmm->N, 1, phmm->M);
}

/*
// InitHMM() This function initializes matrices A, B and vector pi with random values.
//Not doing so can result in the BaumWelch behaving quite weirdly.

void InitHMM(CDHMM *phmm, int N, int M, int seed)
{
	int i, j, k;
	double sum;

	// initialize random number generator

	hmmsetseed(seed);

	phmm->M = M;
	phmm->N = N;

	phmm->pi = (double *)dvector(1, phmm->N);
	sum = 0.0;
	for (i = 1; i <= phmm->N; ++i)
	{
		phmm->pi[i] = hmmgetrand();
		sum += phmm->pi[i];
	}
	for (i = 1; i <= phmm->N; ++i)
		phmm->pi[i] /= sum;

	phmm->A = (double **) dmatrix(1, phmm->N, 1, phmm->N);
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

	phmm->B = (double **) dmatrix(1, phmm->N, 1, phmm->M);
	for (j = 1; j <= phmm->N; ++j)
	{
		sum = 0.0;
		for (k = 1; k <= phmm->M; ++k)
		{
			phmm->B[j][k] = hmmgetrand();
			sum += phmm->B[j][k];
		}
		for (k = 1; k <= phmm->M; ++k)
			phmm->B[j][k] /= sum;
	}
}
*/

void CopyHMM(CDHMM *phmm1, CDHMM *phmm2)
{
	int i, j, k;

	phmm2->M = phmm1->M;
	phmm2->N = phmm1->N;

	phmm2->pi = (double *)dvector(1, phmm2->N);
	for (i = 1; i <= phmm2->N; ++i)
		phmm2->pi[i] = phmm1->pi[i];

	phmm2->A = (double **)dmatrix(1, phmm2->N, 1, phmm2->N);
	for (i = 1; i <= phmm2->N; ++i)
		for (j = 1; j <= phmm2->N; ++j)
			phmm2->A[i][j] = phmm1->A[i][j];

	// FIXME [fix] >>
	//phmm2->params = phmm1->params;
	phmm2->pdf = phmm1->pdf;
}

/*
void PrintHMM(FILE *fp, CDHMM *phmm)
{
	int i, j, k;

	fprintf(fp, "M= %d\n", phmm->M);
	fprintf(fp, "N= %d\n", phmm->N);

	fprintf(fp, "A:\n");
	for (i = 1; i <= phmm->N; ++i)
	{
		for (j = 1; j <= phmm->N; ++j)
			fprintf(fp, "%f ", phmm->A[i][j]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "B:\n");
	for (j = 1; j <= phmm->N; ++j)
	{
		for (k = 1; k <= phmm->M; ++k)
			fprintf(fp, "%f ", phmm->B[j][k]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "pi:\n");
	for (i = 1; i <= phmm->N; ++i)
		fprintf(fp, "%f ", phmm->pi[i]);
	fprintf(fp, "\n\n");
}
*/

double univariate_normal_distribution(const double *symbol, const int state, const void *parameters)
{
	const UnivariateNormalParams *params = reinterpret_cast<const UnivariateNormalParams *>(parameters);

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	boost::math::normal pdf(params->mean, params->stddev);

	boost::math::pdf(pdf, *symbol);
}

double multivariate_normal_distribution(const double *symbol, const int state, const void *parameters)
{
	const UnivariateNormalParams *params = reinterpret_cast<const UnivariateNormalParams *>(parameters);

	//boost::math::normal pdf;  // (default mean = zero, and standard deviation = unity)
	//boost::math::normal pdf(mean, std);

	throw std::runtime_error("not yet implemented");
}

double von_mises_distribution(const double *symbol, const int state, const void *parameters)
{
	const vonMisesParams *params = reinterpret_cast<const vonMisesParams *>(parameters);

	throw std::runtime_error("not yet implemented");
}

double von_mises_fisher_distribution(const double *symbol, const int state, const void *parameters)
{
	const vonMisesFisherParams *params = reinterpret_cast<const vonMisesFisherParams *>(parameters);

	throw std::runtime_error("not yet implemented");
}

}  // umdhmm
