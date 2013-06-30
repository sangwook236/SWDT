//#include "stdafx.h"
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>


namespace {
namespace local {

void gen_rand_data(shogun::SGVector<float64_t> lab, shogun::SGMatrix<float64_t> feat, float64_t dist)
{
	const index_t dims = feat.num_rows;
	const index_t num = lab.vlen;

	for (int32_t i = 0; i < num; ++i)
	{
		if (i < num / 2)
		{
			lab[i] = -1.0;

			for (int32_t j = 0; j < dims; ++j)
				feat(j, i) = shogun::CMath::random(0.0, 1.0) + dist;
		}
		else
		{
			lab[i] = 1.0;

			for (int32_t j = 0; j < dims; ++j)
				feat(j, i) = shogun::CMath::random(0.0, 1.0) - dist;
		}
	}

	lab.display_vector("lab");
	feat.display_matrix("feat");
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/classifier_libsvm.cpp
void classifier_libsvm_example()
{
	const int32_t feature_cache = 0;
	const int32_t kernel_cache = 0;
	const float64_t rbf_width = 10;
	const float64_t svm_C = 10;
	const float64_t svm_eps = 0.001;

	const index_t num = 100;
	const index_t dims = 2;
	const float64_t dist = 0.5;

	shogun::SGVector<float64_t> lab(num);
	shogun::SGMatrix<float64_t> feat(dims, num);

	local::gen_rand_data(lab, feat, dist);

	// create train labels
	shogun::CLabels *labels = new shogun::CBinaryLabels(lab);

	// create train features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(feature_cache);
	SG_REF(features);
	features->set_feature_matrix(feat);

	// create gaussian kernel
	shogun::CGaussianKernel *kernel = new shogun::CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kernel);
	kernel->init(features, features);

	// create svm via libsvm and train
	shogun::CLibSVM *svm = new shogun::CLibSVM(svm_C, kernel, labels);
	SG_REF(svm);
	svm->set_epsilon(svm_eps);
	svm->train();

	SG_SPRINT("num_sv: %d, b: %f\n", svm->get_num_support_vectors(), svm->get_bias());

	// classify + display output
	shogun::CBinaryLabels *out_labels = shogun::CBinaryLabels::obtain_from_generic(svm->apply());

	for (int32_t i = 0; i < num; ++i)
	{
		SG_SPRINT("out[%d] = %f (%f)\n", i, out_labels->get_label(i), out_labels->get_value(i));
	}

	SG_UNREF(out_labels);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(svm);
}

}  // namespace my_shogun
