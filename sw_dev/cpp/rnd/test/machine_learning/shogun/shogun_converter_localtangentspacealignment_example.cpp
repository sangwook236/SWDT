//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/LocalTangentSpaceAlignment.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/converter_localtangentspacealignment.cpp
void converter_localtangentspacealignment_example()
{
	const int N = 100;
	const int dim = 3;
	
	float64_t *matrix = new double [N * dim];
	for (int i = 0; i < N * dim; ++i)
		matrix[i] = shogun::CMath::sin((i / float64_t(N * dim)) * 3.14);

	shogun::CDenseFeatures<double> *features = new shogun::CDenseFeatures<double>(shogun::SGMatrix<double>(matrix, dim, N));
	SG_REF(features);
	
	shogun::CLocalTangentSpaceAlignment *ltsa = new shogun::CLocalTangentSpaceAlignment();
	ltsa->set_target_dim(2);
	ltsa->set_k(4);
	ltsa->parallel->set_num_threads(4);
	shogun::CDenseFeatures<double> *embedding = ltsa->embed(features);
	
	SG_UNREF(embedding);
	SG_UNREF(ltsa);
	SG_UNREF(features);
}

}  // namespace my_shogun
