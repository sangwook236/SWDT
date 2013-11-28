#if !defined(__DYNAMIC_TIME_WARPING__H_)
#define __DYNAMIC_TIME_WARPING__H_ 1


#include <algorithm>
#include <vector>
#include <limits>


/**
 * (c) Daniel Lemire, 2008
 * (c) Earlence Fernandes, Vrije Universiteit Amsterdam 2011
 *
 * This C++ library implements dynamic time warping (DTW). 
 * This library includes the dynamic programming solution for vectored input signals represented
 * by the class Point. Currently, it has 3 dimensions - x, y, z. More can easily be added to this class.
 * No change would be required to the DTW class. Only keep in mind that the distance code has to be updated
 * to accomodate more dimensions.
 *  
 * Time series are represented using STL vectors.
 */

/**
 * N is the length of the time series.
 *
 * maximumWarpingDistance is the maximum warping distance.
 * Typically: maximumWarpingDistance = N / 10.
 * If you set maximumWarpingDistance = N, DTW will be slower.
 *
 * [ref] https://code.google.com/p/lbimproved/
 * [ref] http://en.wikipedia.org/wiki/Dynamic_time_warping
 */
template<typename T, class DistanceMeasure>
double computeFastDynamicTimeWarping(const std::vector<T> &v, const std::vector<T> &w, const std::size_t maximumWarpingDistance, DistanceMeasure distanceMeasure)
{
	const std::size_t N = v.size();
	const std::size_t M = w.size();

	//const std::size_t winSize = std::max(maximumWarpingDistance, N > M ? (N - M) : (M - N));  // adapt window size.
	const std::size_t winSize = std::min(maximumWarpingDistance, std::min(N, M));
	//const std::size_t winSize = maximumWarpingDistance;

	std::vector<std::vector<double> > gamma(N, std::vector<double>(M, std::numeric_limits<double>::max()));

	double bestVal(std::numeric_limits<double>::max());
	for (std::size_t i = 0; i < N; ++i) 
	{
		for (std::size_t j = std::max(0, int(i) - int(winSize)); j < std::min(M, i + winSize + 1); ++j) 
		{
			bestVal = std::numeric_limits<double>::max();
			if (i > 0) 
				bestVal = gamma[i - 1][j];
			if (j > 0) 
				bestVal = std::min(bestVal, gamma[i][j - 1]);
			if (i > 0 && j > 0)
				bestVal = std::min(bestVal, gamma[i - 1][j - 1]);

			gamma[i][j] = (0 == i && 0 == j) ? distanceMeasure(v[i], w[j]) : (bestVal + distanceMeasure(v[i], w[j]));
		}
	}

	return gamma[N-1][M-1];
}


#endif  // __DYNAMIC_TIME_WARPING__H_
