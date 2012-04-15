/*
**	Author: Tapas Kanungo, kanungo@cfar.umd.edu
**	File:	hmmrand.c
**	Date:	4 May 1999
**	Purpose: To separate out the random number generator
** 		functions so that the rest of the code can be
**		platform independent.
*/

#include "umdhmm_nrutil.h"
#include <sys/types.h>
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
#include <unistd.h>
#else
#include <ctime>
#endif
#include <cstdlib>


namespace umdhmm {

/*
** hmmgetseed() generates an arbitary seed for the random number generator.
*/
int  hmmgetseed(void) 
{
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
	return ((int)getpid());
#else
	return (int)std::time(NULL);
#endif
}

/* 
** hmmsetseed() sets the seed of the random number generator to a
** specific value.
*/
void hmmsetseed(int seed) 
{
	std::srand(seed);
}

/*
**  hmmgetrand() returns a (double) pseudo random number in the
**  interval [0,1).
*/
double hmmgetrand(void)
{
	return (double)std::rand() / RAND_MAX;
}

double hmmgetrand(double lb, double ub)
{
	return ((double)std::rand() / RAND_MAX) * (ub - lb) + lb;
}

}  // namespace umdhmm
