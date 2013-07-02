//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/converter_kernellocallylinearembedding.cpp
void converter_kernellocallylinearembedding_example()
{
	const int N = 100;
	const int dim = 3;

	float64_t *matrix = new double [N * dim];
	for (int i = 0; i < N * dim; ++i)
		matrix[i] = shogun::CMath::sin((i / float64_t(N * dim)) * 3.14);

	shogun::CDenseFeatures<double> *features = new shogun::CDenseFeatures<double>(shogun::SGMatrix<double>(matrix, dim, N));
	SG_REF(features);

	shogun::CKernelLocallyLinearEmbedding *klle = new shogun::CKernelLocallyLinearEmbedding();
	shogun::CKernel *kernel = new shogun::CLinearKernel();
	klle->set_target_dim(2);
	klle->set_k(4);
	klle->set_kernel(kernel);
	klle->parallel->set_num_threads(4);
	shogun::CDenseFeatures<double> *embedding = klle->embed(features);

    // show result.
	embedding->get_feature_matrix().display_matrix("kernel locally linear embedding (LLE) - result");

	SG_UNREF(embedding);
	SG_UNREF(klle);
	SG_UNREF(features);
}

}  // namespace my_shogun
