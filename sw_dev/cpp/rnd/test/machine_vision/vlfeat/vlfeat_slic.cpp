#include <vl/slic.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

// simple linear iterative clustering (SLIC)
// superpixel extraction (segmentation) method based on a local version of k-means
void slic()
{
	vl_uint32 *segmentation = NULL;
	float const *image = NULL;
	vl_size width;
	vl_size height;
	vl_size numChannels;

	// TODO [implement] >>

	//
	const vl_size regionSize = 10;
	const float regularization = 1.0f;
	const vl_size minRegionSize = 1;
	vl_slic_segment(segmentation, image, width, height, numChannels, regionSize, regularization, minRegionSize);
}

}  // namespace my_vlfeat
