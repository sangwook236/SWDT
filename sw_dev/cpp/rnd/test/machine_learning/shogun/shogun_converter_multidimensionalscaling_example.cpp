//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/converter_multidimensionalscaling.cpp
void converter_multidimensionalscaling_example()
{
	const int N = 100;
	const int dim = 3;
	
	float64_t *matrix = new double [N * dim];
	for (int i = 0; i < N * dim; ++i)
		matrix[i] = shogun::CMath::sin((i / float64_t(N * dim)) * 3.14);

	shogun::CDenseFeatures<double> *features = new shogun::CDenseFeatures<double>(shogun::SGMatrix<double>(matrix, dim, N));
	SG_REF(features);
	shogun::CMultidimensionalScaling *mds = new shogun::CMultidimensionalScaling();
	mds->set_target_dim(2);
	mds->set_landmark(true);
	mds->parallel->set_num_threads(4);
	shogun::CDenseFeatures<double> *embedding = mds->embed(features);
	
	SG_UNREF(embedding);
	SG_UNREF(mds);
	SG_UNREF(features);
}

}  // namespace my_shogun
