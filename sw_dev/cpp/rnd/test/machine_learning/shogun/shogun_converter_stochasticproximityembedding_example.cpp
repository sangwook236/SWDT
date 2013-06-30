//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/converter_stochasticproximityembedding.cpp
void converter_stochasticproximityembedding_example()
{
	const int N = 100;
	const int dim = 3;

	// Generate toy data
	shogun::SGMatrix<float64_t> matrix(dim, N);
	for (int i =0 ; i < N * dim; ++i)
		matrix[i] = shogun::CMath::sin((i / float64_t(N * dim)) * 3.14);

	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(matrix);
	SG_REF(features);

	// Create embedding and set parameters for global strategy
	shogun::CStochasticProximityEmbedding *spe = new shogun::CStochasticProximityEmbedding();
	spe->set_target_dim(2);
	spe->set_strategy(SPE_GLOBAL);
	spe->set_nupdates(40);
	SG_REF(spe);

	// Apply embedding with global strategy
	shogun::CDenseFeatures<float64_t> *embedding = spe->embed(features);
	SG_REF(embedding);

	// Set parameters for local strategy
	spe->set_strategy(SPE_LOCAL);
	spe->set_k(12);

	// Apply embedding with local strategy
	SG_UNREF(embedding);
	embedding = spe->embed(features);
	SG_REF(embedding);

	// Free memory
	SG_UNREF(embedding);
	SG_UNREF(spe);
	SG_UNREF(features);
}

}  // namespace my_shogun
