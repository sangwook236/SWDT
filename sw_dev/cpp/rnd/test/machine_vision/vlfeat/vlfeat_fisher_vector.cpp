#include <vl/fisher.h>
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <string>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// [ref] ${VLFEAT_HOME}/src/test_gmm.c.
// [ref] http://www.vlfeat.org/api/fisher.html.
void fisher_vector()
{
/*
	// create a GMM object and cluster input data to get means, covariances and priors of the estimated mixture.
	VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

	vl_gmm_cluster(gmm, dataToEncode, numDataToEncode);

	// allocate space for the encoding.
	float *enc = (float *)vl_malloc(sizeof(float) * 2 * dimension * numClusters);

	// run fisher encoding.
	vl_fisher_encode(
		enc, VL_TYPE_FLOAT,
		vl_gmm_get_means(gmm), dimension, numClusters,
		vl_gmm_get_covariances(gmm),
		vl_gmm_get_priors(gmm),
		dataToEncode, numDataToEncode,
		VL_FISHER_FLAG_IMPROVED
	);
*/
}

}  // namespace my_vlfeat
