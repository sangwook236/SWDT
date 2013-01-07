/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   22 February 1998
**      File:   sequence.c
**      Purpose: Routines for generating, reading and writing sequence of
**		observation symbols.
**      Organization: University of Maryland
**
**	Update:
**	Author: Tapas Kanungo
**	Purpose: To make calls to generic random number generators
**		and to change the seed everytime the software is executed.
**
**      $Id: sequence.c,v 1.2 1998/02/23 06:19:41 kanungo Exp kanungo $
*/

#include "umdhmm_nrutil.h"
#include "umdhmm_cdhmm.h"
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>


namespace umdhmm {

static char rcsid[] = "$Id: sequence.c,v 1.2 1998/02/23 06:19:41 kanungo Exp kanungo $";

int GenInitialState(CDHMM *phmm);
int GenNextState(CDHMM *phmm, int q_t);
void GenSymbol_UnivariateNormal(CDHMM *phmm, boost::minstd_rand &baseGenerator, int q_t, double *o_t);

void GenSequenceArray_UnivariateNormal(CDHMM *phmm, int seed, int T, double **O, int *q)
{
	hmmsetseed(seed);

	boost::minstd_rand baseGenerator(seed);

	q[1] = GenInitialState(phmm);
	GenSymbol_UnivariateNormal(phmm, baseGenerator, q[1], O[1]);

	for (int t = 2; t <= T; ++t)
	{
		q[t] = GenNextState(phmm, q[t-1]);
		GenSymbol_UnivariateNormal(phmm, baseGenerator, q[t], O[t]);
	}
}

int GenInitialState(CDHMM *phmm)
{
	double val = hmmgetrand();
	double accum = 0.0;
	int q_t = phmm->N;
	for (int i = 1; i <= phmm->N; ++i)
	{
		if (val < phmm->pi[i] + accum)
		{
			q_t = i;
			break;
		}
		else
			accum += phmm->pi[i];
	}

	return q_t;
}

int GenNextState(CDHMM *phmm, int q_t)
{
	double val = hmmgetrand();
	double accum = 0.0;
	int q_next = phmm->N;
	for (int j = 1; j <= phmm->N; ++j)
	{
		if (val < phmm->A[q_t][j] + accum)
		{
			q_next = j;
			break;
		}
		else
			accum += phmm->A[q_t][j];
	}

	return q_next;
}

void GenSymbol_UnivariateNormal(CDHMM *phmm, boost::minstd_rand &baseGenerator, int q_t, double *o_t)
{
	typedef boost::minstd_rand base_generator_type;
	typedef boost::normal_distribution<> distribution_type;
	typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

	const umdhmm::UnivariateNormalParams *set_of_params = reinterpret_cast<umdhmm::UnivariateNormalParams *>(phmm->set_of_params);
	generator_type normal_gen(baseGenerator, distribution_type(set_of_params[q_t].mean, set_of_params[q_t].stddev));
	for (int j = 1; j <= phmm->M; ++j)
		o_t[j] = normal_gen();
}

void ReadSequence(FILE *fp, int *pT, int *pM, double ***pO)
{
	fscanf(fp, "T= %d\n", pT);
	fscanf(fp, "M= %d\n", pM);
	double **O = dmatrix(1, *pT, 1, *pM);
	for (int i = 1; i <= *pT; ++i)
		for (int j = 1; j <= *pM; ++j)
			fscanf(fp, "%lf", &O[i][j]);
	*pO = O;
}

void PrintSequence(FILE *fp, int T, int M, double **O)
{
	fprintf(fp, "T= %d\n", T);
	fprintf(fp, "M= %d\n", M);
	for (int i = 1; i <= T; ++i)
	{
		for (int j = 1; j <= M; ++j)
			fprintf(fp, "%lf ", O[i][j]);
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
}

}  // namespace umdhmm
