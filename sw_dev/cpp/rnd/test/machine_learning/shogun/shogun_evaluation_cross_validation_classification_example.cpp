//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/evaluation_cross_validation_classification.cpp
void evaluation_cross_validation_classification_example()
{
	// data matrix dimensions
	const index_t num_vectors = 40;
	const index_t num_features = 5;

	// data means -1, 1 in all components, std deviation of 3
	shogun::SGVector<float64_t> mean_1(num_features);
	shogun::SGVector<float64_t> mean_2(num_features);
	shogun::SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -1.0);
	shogun::SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 1.0);
	float64_t sigma = 3;

	shogun::SGVector<float64_t>::display_vector(mean_1.vector, mean_1.vlen, "mean 1");
	shogun::SGVector<float64_t>::display_vector(mean_2.vector, mean_2.vlen, "mean 2");

	// fill data matrix around mean
	shogun::SGMatrix<float64_t> train_dat(num_features, num_vectors);
	for (index_t i = 0; i < num_vectors; ++i)
	{
		for (index_t j = 0; j < num_features; ++j)
		{
			const float64_t mean = i < num_vectors / 2 ? mean_1.vector[0] : mean_2.vector[0];
			train_dat.matrix[i * num_features + j] = shogun::CMath::normal_random(mean, sigma);
		}
	}

	// training features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	// training labels +/- 1 for each cluster
	shogun::SGVector<float64_t> lab(num_vectors);
	for (index_t i = 0; i < num_vectors; ++i)
		lab.vector[i] = i < num_vectors / 2 ? -1.0 : 1.0;

	shogun::CBinaryLabels *labels = new shogun::CBinaryLabels(lab);

	// Gaussian kernel
	const int32_t kernel_cache = 100;
	const int32_t width = 10;
	shogun::CGaussianKernel *kernel = new shogun::CGaussianKernel(kernel_cache, width);
	kernel->init(features, features);

	// create svm via libsvm
	const float64_t svm_C = 10;
	const float64_t svm_eps = 0.0001;
	shogun::CLibSVM *svm = new CLibSVM(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	// train and output
	svm->train(features);
	shogun::CBinaryLabels *output = shogun::CBinaryLabels::obtain_from_generic(svm->apply(features));
	for (index_t i = 0; i < num_vectors; ++i)
		SG_SPRINT("i = %d, class = %f,\n", i, output->get_label(i));

	// evaluation criterion
	shogun::CContingencyTableEvaluation *eval_crit = new shogun::CContingencyTableEvaluation(ACCURACY);

	// evaluate training error
	float64_t eval_result = eval_crit->evaluate(output, labels);
	SG_SPRINT("training error: %f\n", eval_result);
	SG_UNREF(output);

	// assert that regression "works". this is not guaranteed to always work
	// but should be a really coarse check to see if everything is going approx. right
	ASSERT(eval_result < 2);

	// splitting strategy
	const index_t n_folds = 5;
	shogun::CStratifiedCrossValidationSplitting *splitting = new shogun::CStratifiedCrossValidationSplitting(labels, n_folds);

	// cross validation instance, 10 runs, 95% confidence interval
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(svm, features, labels, splitting, eval_crit);

	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.05);

	// actual evaluation
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	result->print_result();

	// clean up
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
}

}  // namespace my_shogun
