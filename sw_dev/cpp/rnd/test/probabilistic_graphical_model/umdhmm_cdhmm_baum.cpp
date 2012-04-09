/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   baumwelch.c
**      Purpose: Baum-Welch algorithm for estimating the parameters
**              of a CDHMM model, given an observation sequence.
**      Organization: University of Maryland
**
**	Update:
**	Author: Tapas Kanungo
**	Date:	19 April 1999
**	Purpose: Changed the convergence criterion from ratio
**		to absolute value.
**
**      $Id: baumwelch.c,v 1.6 1999/04/24 15:58:43 kanungo Exp kanungo $
*/

#include "umdhmm_nrutil.h"
#include "umdhmm_cdhmm.h"
#include "umdhmm_hmm.h"
#include <cmath>


namespace umdhmm {

static char rcsid[] = "$Id: baumwelch.c,v 1.6 1999/04/24 15:58:43 kanungo Exp kanungo $";

void BaumWelch_UnivariateNormal(CDHMM *phmm, int T, double **O, const double tol, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal)
{
	double ***xi = AllocXi(T, phmm->N);
	double *scale = dvector(1, T);

	double logprobf, logprobb;
	ForwardWithScale(phmm, T, O, alpha, scale, &logprobf);
	BackwardWithScale(phmm, T, O, beta, scale, &logprobb);
	ComputeGamma(phmm, T, alpha, beta, gamma);
	ComputeXi(phmm, T, O, alpha, beta, xi);

	*plogprobinit = logprobf;  // log P(O | initial model)
	double logprobprev = logprobf;

	umdhmm::UnivariateNormalParams *set_of_params = (umdhmm::UnivariateNormalParams *)phmm->set_of_params;

	double numeratorA, denominatorA;
	double numeratorP, denominatorP;
	double delta; //, deltaprev = 10e-70;

	int l = 0;
	int	i, j;
	int	t;
	do
	{
		for (i = 1; i <= phmm->N; ++i)
		{
			// reestimate frequency of state i in time t=1
			phmm->pi[i] = .001 + .999 * gamma[1][i];

			// reestimate transition matrix 
			denominatorA = 0.0;
			for (t = 1; t <= T - 1; ++t)
				denominatorA += gamma[t][i];

			for (j = 1; j <= phmm->N; ++j)
			{
				numeratorA = 0.0;
				for (t = 1; t <= T - 1; ++t)
					numeratorA += xi[t][i][j];
				phmm->A[i][j] = .001 + .999 * numeratorA / denominatorA;
			}

			// reestimate symbol prob in each state
			denominatorP = denominatorA + gamma[T][i];
#if 0
			// for multivariate normal distributions
			for (k = 1; k <= phmm->M; ++k)
			{
				numeratorP = 0.0;
				for (t = 1; t <= T; ++t)
				{
					numeratorP += gamma[t][i] * O[t][k];
				}
				set_of_params[i].mean[k] = .001 + .999 * numeratorP / denominatorP;
			}
#else
			// for univariate normal distributions
			numeratorP = 0.0;
			for (t = 1; t <= T; ++t)
			{
				numeratorP += gamma[t][i] * O[t][1];
			}
			set_of_params[i].mean = .001 + .999 * numeratorP / denominatorP;
#endif

#if 0
			// for multivariate normal distributions
			for (k = 1; k <= phmm->M; ++k)
			{
				numeratorP = 0.0;
				for (t = 1; t <= T; ++t)
				{
					numeratorP += gamma[t][i] * (O[t][k] - set_of_params[i].mean[k]) * (O[t][k] - set_of_params[i].mean[k]);
				}
				set_of_params[i].stddev[k] = .001 + .999 * numeratorP / denominatorP;
			}
#else
			// for univariate normal distributions
			numeratorP = 0.0;
			for (t = 1; t <= T; ++t)
			{
				numeratorP += gamma[t][i] * (O[t][1] - set_of_params[i].mean) * (O[t][1] - set_of_params[i].mean);
			}
			set_of_params[i].stddev = .001 + .999 * numeratorP / denominatorP;
#endif
		}

		ForwardWithScale(phmm, T, O, alpha, scale, &logprobf);
		BackwardWithScale(phmm, T, O, beta, scale, &logprobb);
		ComputeGamma(phmm, T, alpha, beta, gamma);
		ComputeXi(phmm, T, O, alpha, beta, xi);

		// compute difference between log probability of two iterations
		delta = logprobf - logprobprev;
		logprobprev = logprobf;
		++l;
	} while (delta > tol);  // if log probability does not change much, exit

	*pniter = l;
	*plogprobfinal = logprobf;  // log P(O | estimated model)
	FreeXi(xi, T, phmm->N);
	free_dvector(scale, 1, T);
}

void ComputeGamma(CDHMM *phmm, int T, double **alpha, double **beta, double **gamma)
{
	int i, j;
	int	t;
	double denominator;

	for (t = 1; t <= T; ++t)
	{
		denominator = 0.0;
		for (j = 1; j <= phmm->N; ++j)
		{
			gamma[t][j] = alpha[t][j] * beta[t][j];
			denominator += gamma[t][j];
		}

		for (i = 1; i <= phmm->N; ++i)
			gamma[t][i] = gamma[t][i] / denominator;
	}
}

void ComputeXi(CDHMM* phmm, int T, double **O, double **alpha, double **beta, double ***xi)
{
	int i, j;
	int t;
	double sum;

	for (t = 1; t <= T - 1; ++t)
	{
		sum = 0.0;
		for (i = 1; i <= phmm->N; ++i)
			for (j = 1; j <= phmm->N; ++j)
			{
				//xi[t][i][j] = alpha[t][i] * beta[t+1][j] * (phmm->A[i][j]) * (phmm->B[j][O[t+1]]);
				xi[t][i][j] = alpha[t][i] * beta[t+1][j] * (phmm->A[i][j]) * (phmm->pdf(O[t+1], j, phmm->set_of_params));
				sum += xi[t][i][j];
			}

		for (i = 1; i <= phmm->N; ++i)
			for (j = 1; j <= phmm->N; ++j)
				xi[t][i][j] /= sum;
	}
}

}  // namespace umdhmm
