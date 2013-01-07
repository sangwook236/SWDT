/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   hmm.h
**      Purpose: datastructures used for CDHMM.
**      Organization: University of Maryland
**
**	Update:
**	Author: Tapas Kanungo
**	Purpose: include <math.h>. Not including this was
**		creating a problem with forward.c
**      $Id: hmm.h,v 1.9 1999/05/02 18:38:11 kanungo Exp kanungo $
*/

#if !defined(__umdhmm_cdhmm_h__)
#define __umdhmm_cdhmm_h__ 1

#include <cstdlib>
#include <cstdio>
#include <cmath>


namespace umdhmm {

struct CDHMM
{
	int N;  // number of hidden states; Q={1,2,...,N}
	int M;  // number of observation symbols; V={1,2,...,M}
	double *pi;  // pi[1..N] pi[i] is the initial state distribution.
	double **A;  // A[1..N][1..N]. a[i][j] is the transition prob. of going from state i at time t to state j at time t+1
	void *set_of_params;  // parameters of the observation (emission) probability 

	// if state == 1, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 2, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N, hidden state = [ 0 0 0 ... 0 1 ]
	double (*pdf)(const double *symbol, const int state, const void *params);  // the observation (emission) probability of observing symbol in state
};

void ReadCDHMM_UnivariateNormal(FILE *fp, CDHMM *phmm);
void PrintCDHMM_UnivariateNormal(FILE *fp, CDHMM *phmm);
void InitCDHMM_UnivariateNormal(CDHMM *phmm, int N, int M, int seed);
void CopyCDHMM_UnivariateNormal(CDHMM *phmm1, CDHMM *phmm2);
void FreeCDHMM_UnivariateNormal(CDHMM *phmm);

void ReadSequence(FILE *fp, int *pT, int *pM, double ***pO);
void PrintSequence(FILE *fp, int T, int M, double **O);
void GenSequenceArray_UnivariateNormal(CDHMM *phmm, int seed, int T, double **O, int *q);

void Forward(CDHMM *phmm, int T, double **O, double **alpha, double *pprob);
void ForwardWithScale(CDHMM *phmm, int T, double **O, double **alpha, double *scale, double *pprob);
void Backward(CDHMM *phmm, int T, double **O, double **beta, double *pprob);
void BackwardWithScale(CDHMM *phmm, int T, double **O, double **beta, double *scale, double *pprob);
void BaumWelch_UnivariateNormal(CDHMM *phmm, int T, double **O, const double tol, double **alpha, double **beta, double **gamma, int *niter, double *plogprobinit, double *plogprobfinal);
void Viterbi(CDHMM *phmm, int T, double **O, double **delta, int **psi, int *q, double *pprob);
void ViterbiLog(CDHMM *phmm, int T, double **O, double **delta, int **psi, int *q, double *pprob);

void ComputeGamma(CDHMM *phmm, int T, double **alpha, double **beta, double **gamma);
void ComputeXi(CDHMM* phmm, int T, double **O, double **alpha, double **beta, double ***xi);


struct UnivariateNormalParams
{
	double mean;
	double stddev;
};

struct MultivariateNormalParams
{
	//size_t dim;  // the number of observation symbols
	double *mean;
	double *covar;
};

struct vonMisesParams
{
	double mean;
	double kappa;
};

struct vonMisesFisherParams
{
	//size_t dim;  // the number of observation symbols
	double *mean;
	double kappa;
};

double univariate_normal_distribution(const double *symbol, const int state, const void *set_of_parameters);
double multivariate_normal_distribution(const double *symbol, const int state, const void *set_of_parameters);
double von_mises_distribution(const double *symbol, const int state, const void *set_of_parameters);
double von_mises_fisher_distribution(const double *symbol, const int state, const void *set_of_parameters);

UnivariateNormalParams * AllocSetOfParams_UnivariateNormal(int nl, int nh);
void FreeSetOfParams_UnivariateNormal(void *s, int nl, int nh);

}  // namespace umdhmm


#endif  // __umdhmm_cdhmm_h__
