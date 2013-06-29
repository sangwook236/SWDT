//#include "stdafx.h"
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/base/init.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/classifier_multiclasslibsvm.cpp
void classifier_multiclasslibsvm_example()
{
	const index_t num_vec = 3;
	const index_t num_feat = 2;
	const index_t num_class = 2;

	// create some data
	shogun::SGMatrix<float64_t> matrix(num_feat, num_vec);
	shogun::SGVector<float64_t>::range_fill_vector(matrix.matrix, num_feat * num_vec);

	// create vectors
	// shogun will now own the matrix created
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(matrix);

	// create three labels
	shogun::CMulticlassLabels *labels = new shogun::CMulticlassLabels(num_vec);
	for (index_t i = 0; i < num_vec; ++i)
		labels->set_label(i, i % num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	shogun::CGaussianKernel *kernel = new shogun::CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	shogun::CMulticlassLibSVM *svm = new shogun::CMulticlassLibSVM(10, kernel, labels);
	svm->train();

	// classify on training examples
	shogun::CMulticlassLabels *output = shogun::CMulticlassLabels::obtain_from_generic(svm->apply());
	shogun::SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "batch output");

	// assert that batch apply and apply(index_t) give same result
	for (index_t i = 0; i < output->get_num_labels(); ++i)
	{
		const float64_t label = svm->apply_one(i);
		SG_SPRINT("single output[%d] = %f\n", i, label);
		ASSERT(output->get_label(i)==label);
	}
	SG_UNREF(output);

	// free up memory
	SG_UNREF(svm);
}

}  // namespace my_shogun
