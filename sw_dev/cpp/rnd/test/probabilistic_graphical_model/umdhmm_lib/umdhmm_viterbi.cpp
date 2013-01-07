/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   viterbi.c
**      Purpose: Viterbi algorithm for computing the maximum likelihood
**		state sequence and probablity of observing a sequence
**		given the model.
**      Organization: University of Maryland
**
**      $Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $
*/

#include "umdhmm_hmm.h"
#include "umdhmm_nrutil.h"
#include <limits>
#include <cmath>


namespace umdhmm {

static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

void Viterbi(HMM *phmm, int T, int *O, double **delta, int **psi, int *q, double *pprob)
{
	int 	i, j;	/* state indices */
	int  	t;	/* time index */

	int	maxvalind;
	double maxval, val;

	/* 1. Initialization  */

	for (i = 1; i <= phmm->N; ++i)
	{
		delta[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]);
		psi[1][i] = 0;
	}

	/* 2. Recursion */

	for (t = 2; t <= T; ++t)
	{
		for (j = 1; j <= phmm->N; ++j)
		{
			maxval = 0.0;
			maxvalind = 1;
			for (i = 1; i <= phmm->N; ++i)
			{
				val = delta[t-1][i] * (phmm->A[i][j]);
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			delta[t][j] = maxval * (phmm->B[j][O[t]]);
			psi[t][j] = maxvalind;
		}
	}

	/* 3. Termination */

	*pprob = 0.0;
	q[T] = 1;
	for (i = 1; i <= phmm->N; ++i)
	{
		if (delta[T][i] > *pprob)
		{
			*pprob = delta[T][i];
			q[T] = i;
		}
	}

	/* 4. Path (state sequence) backtracking */

	for (t = T - 1; t >= 1; --t)
		q[t] = psi[t+1][q[t+1]];
}

void ViterbiLog(HMM *phmm, int T, int *O, double **delta, int **psi, int *q, double *pprob)
{
	int     i, j;   /* state indices */
	int     t;      /* time index */

	int     maxvalind;
	double  maxval, val;
	double  **biot;

	/* 0. Preprocessing */

	for (i = 1; i <= phmm->N; ++i)
		phmm->pi[i] = std::log(phmm->pi[i]);
	for (i = 1; i <= phmm->N; ++i)
		for (j = 1; j <= phmm->N; ++j)
		{
			phmm->A[i][j] = std::log(phmm->A[i][j]);
		}

	biot = dmatrix(1, phmm->N, 1, T);
	for (i = 1; i <= phmm->N; ++i)
		for (t = 1; t <= T; ++t)
		{
			biot[i][t] = std::log(phmm->B[i][O[t]]);
		}

	/* 1. Initialization  */

	for (i = 1; i <= phmm->N; ++i)
	{
		delta[1][i] = phmm->pi[i] + biot[i][1];
		psi[1][i] = 0;
	}

	/* 2. Recursion */

	for (t = 2; t <= T; ++t)
	{
		for (j = 1; j <= phmm->N; ++j)
		{
			maxval = -std::numeric_limits<double>::max();
			maxvalind = 1;
			for (i = 1; i <= phmm->N; ++i)
			{
				val = delta[t-1][i] + (phmm->A[i][j]);
				if (val > maxval)
				{
					maxval = val;
					maxvalind = i;
				}
			}

			delta[t][j] = maxval + biot[j][t];
			psi[t][j] = maxvalind;
		}
	}

	/* 3. Termination */

	*pprob = -std::numeric_limits<double>::max();
	q[T] = 1;
	for (i = 1; i <= phmm->N; ++i)
	{
		if (delta[T][i] > *pprob)
		{
			*pprob = delta[T][i];
			q[T] = i;
		}
	}

	/* 4. Path (state sequence) backtracking */

	for (t = T - 1; t >= 1; --t)
		q[t] = psi[t+1][q[t+1]];
}

}  // namespace umdhmm
