#include <vl/liop.h>
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/imopv.h>
#include <cstdio>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// [ref] ${VLFEAT_HOME}/src/test_liop.c.
// [ref] http://www.vlfeat.org/api/liop.html.
void liop()
{
/*
	// create a new object instance (these numbers corresponds to parameter values proposed by authors of the paper, except for 41).
	const vl_size sideLength = 41;
	VlLiopDesc *liop = vl_liopdesc_new_basic(sideLength);

	// allocate the descriptor array.
	const vl_size dimension = vl_liopdesc_get_dimension(liop);
	float *desc = (float *)vl_malloc(sizeof(float) * dimension);

	const float *patch = ...;

	// the threshold is used to discard low-contrast oder pattern in the computation of the statistics. 
	const float threshold = ...;
	vl_liopdesc_set_intensity_threshold(liop, threshold);

	// compute descriptor from a patch (an array of length sideLegnth * sideLength).
	vl_liopdesc_process(liop, desc, patch);

	// delete the object.
	vl_liopdesc_delete(liop);
*/

	const vl_size size = 11 * 11;
	float mat[] = {
		6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
		6, 6, 6, 5, 4, 4, 4, 5, 6, 6, 6,
		6, 6, 5, 4, 3, 3, 3, 4, 5, 6, 6,
		6, 5, 4, 3, 2, 2, 2, 3, 4, 5, 6,
		6, 4, 3, 2, 2, 1, 2, 2, 3, 4, 6,
		6, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6,
		6, 4, 3, 2, 2, 1, 2, 2, 3, 4, 6,
		6, 5, 4, 3, 2, 2, 2, 3, 4, 5, 6,
		6, 6, 5, 4, 3, 3, 3, 4, 5, 6, 6,
		6, 6, 6, 5, 4, 4, 4, 5, 6, 6, 6,
		6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
	};

	float *patch = (float *)vl_malloc(sizeof(float) * size);

	for (vl_int i = 0; i < (signed)size; ++i)
	{
		patch[i] = mat[i];
	}

	VlLiopDesc *liop = vl_liopdesc_new(4, 6, 2, 11);

	vl_liopdesc_delete(liop) ;
	vl_free(patch) ;
}

}  // namespace my_vlfeat
