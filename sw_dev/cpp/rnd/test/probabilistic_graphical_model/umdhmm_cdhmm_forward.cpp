/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   forward.c
**      Purpose: Foward algorithm for computing the probabilty
**		of observing a sequence given a CDHMM model parameter.
**      Organization: University of Maryland
**
**      $Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $
*/
#include "umdhmm_cdhmm.h"
#include <iostream>


namespace umdhmm {

static char rcsid[] = "$Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $";

void Forward(CDHMM *phmm, int T, double **O, double **alpha, double *pprob)
{
	int i, j;  // state indices
	int t;  // time index

	// 1. Initialization
	for (i = 1; i <= phmm->N; ++i)
		//alpha[1][i] = phmm->pi[i] * phmm->B[i][O[1]];
		alpha[1][i] = phmm->pi[i] * (phmm->pdf(O[1], i, phmm->set_of_params));

	// 2. Induction
	double sum;  // partial sum
	for (t = 1; t < T; ++t)
	{
		for (j = 1; j <= phmm->N; ++j)
		{
			sum = 0.0;
			for (i = 1; i <= phmm->N; ++i)
				sum += alpha[t][i] * (phmm->A[i][j]);

			//alpha[t+1][j] = sum * (phmm->B[j][O[t+1]]);
			alpha[t+1][j] = sum * (phmm->pdf(O[t+1], j, phmm->set_of_params));
		}
	}

	// 3. Termination
	*pprob = 0.0;
	for (i = 1; i <= phmm->N; ++i)
		*pprob += alpha[T][i];
}

void ForwardWithScale(CDHMM *phmm, int T, double **O, double **alpha, double *scale, double *pprob)
// pprob is the LOG probability
{
	int	i, j;  // state indices
	int	t;	// time index

	// 1. Initialization
	scale[1] = 0.0;
	for (i = 1; i <= phmm->N; ++i)
	{
		//alpha[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]);
		alpha[1][i] = phmm->pi[i] * (phmm->pdf(O[1], i, phmm->set_of_params));
		scale[1] += alpha[1][i];
	}
	for (i = 1; i <= phmm->N; ++i)
		alpha[1][i] /= scale[1];

	// 2. Induction
	double sum;  // partial sum
	for (t = 1; t <= T - 1; ++t)
	{
		scale[t+1] = 0.0;
		for (j = 1; j <= phmm->N; ++j)
		{
			sum = 0.0;
			for (i = 1; i <= phmm->N; ++i)
				sum += alpha[t][i] * (phmm->A[i][j]);

			//alpha[t+1][j] = sum * (phmm->B[j][O[t+1]]);
			alpha[t+1][j] = sum * (phmm->pdf(O[t+1], j, phmm->set_of_params));
			scale[t+1] += alpha[t+1][j];
		}
		for (j = 1; j <= phmm->N; ++j)
			alpha[t+1][j] /= scale[t+1];
	}

	// 3. Termination
	*pprob = 0.0;
	for (t = 1; t <= T; ++t)
		*pprob += std::log(scale[t]);
}

}  // namespace umdhmm
