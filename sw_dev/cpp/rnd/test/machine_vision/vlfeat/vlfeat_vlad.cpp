#include <vl/vlad.h>
#include <vl/gmm.h>
#include <vl/kmeans.h>
#include <vl/mathop.h>
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// [ref] ${VLFEAT_HOME}/src/test_gmm.c.
// [ref] http://www.vlfeat.org/api/vlad.html.
void vlad()
{
/*
	// create a KMeans object and run clustering to get vocabulary words (centers).
	VlKMeans *kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VLDistanceL2);
	vl_kmeans_cluster(kmeans, dataToEncode, dimension, numDataToEncode, numCenters);

	// find nearest cliuster centers for the data that should be encoded.
	vl_uint32 *indexes = (vl_uint32 *)vl_malloc(sizeof(vl_uint32) * numDataToEncode);
	vl_kmeans_quantize(kmeans, indexes, dataToEncode, numDataToEncode);

	// convert indexes array to assignments array, which can be processed by vl_vlad_encode.
	float *assignments = (float *)vl_malloc(sizeof(float) * numDataToEncode * numCenters);
	memset(assignments, 0, sizeof(float) * numDataToEncode * numCenters);
	for (int i = 0; i < numDataToEncode; ++i)
	{
		assignments[i + numDataToEncode * indexes[i]] = 1.;
	}

	// allocate space for vlad encoding.
	float *enc = (float *)vl_malloc(sizeof(VL_TYPE_FLOAT) * dimension * numCenters);

	// do the encoding job.
	vl_vlad_encode(
		enc, VL_TYPE_FLOAT,
		vl_kmeans_get_centers(kmeans), dimension, numCenters,
		dataToEncode, numDataToEncode,
		assignments,
		0
	);
*/
}

}  // namespace my_vlfeat
