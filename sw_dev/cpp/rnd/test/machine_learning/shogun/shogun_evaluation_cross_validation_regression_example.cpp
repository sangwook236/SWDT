//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanSquaredError.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/evaluation_cross_validation_regression.cpp
void evaluation_cross_validation_regression_example()
{
	// data matrix dimensions
	const index_t num_vectors = 100;
	const index_t num_features = 1;

	// training label data
	shogun::SGVector<float64_t> lab(num_vectors);

	// fill data matrix and labels
	shogun::SGMatrix<float64_t> train_dat(num_features, num_vectors);
	shogun::SGVector<float64_t>::range_fill_vector(train_dat.matrix, num_vectors);
	for (index_t i = 0; i < num_vectors; ++i)
	{
		// labels are linear plus noise
		lab.vector[i] = i + shogun::CMath::normal_random(0, 1.0);

	}

	// training features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	// training labels
	shogun::CRegressionLabels *labels=new shogun::CRegressionLabels(lab);

	// kernel
	shogun::CLinearKernel *kernel = new shogun::CLinearKernel();
	kernel->init(features, features);

	// kernel ridge regression
	const float64_t tau = 0.0001;
	shogun::CKernelRidgeRegression *krr = new shogun::CKernelRidgeRegression(tau, kernel, labels);

	// evaluation criterion
	shogun::CMeanSquaredError *eval_crit = new shogun::CMeanSquaredError();

	// train and output
	krr->train(features);
	shogun::CRegressionLabels *output = shogun::CRegressionLabels::obtain_from_generic(krr->apply());
	for (index_t i = 0; i < num_vectors; ++i)
	{
		SG_SPRINT("x = %f, train = %f, predict = %f\n", train_dat.matrix[i], labels->get_label(i), output->get_label(i));
	}

	// evaluate training error
	const float64_t eval_result = eval_crit->evaluate(output, labels);
	SG_SPRINT("training error: %f\n", eval_result);
	SG_UNREF(output);

	// assert that regression "works".
	// this is not guaranteed to always work but should be a really coarse check to see if everything is going approx. right
	ASSERT(eval_result < 2);

	// splitting strategy
	const index_t n_folds = 5;
	shogun::CCrossValidationSplitting *splitting = new shogun::CCrossValidationSplitting(labels, n_folds);

	// cross validation instance, 10 runs, 95% confidence interval
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(krr, features, labels, splitting, eval_crit);

	cross->set_num_runs(100);
	cross->set_conf_int_alpha(0.05);

	// actual evaluation
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("cross_validation estimate:\n");
	result->print_result();

	// ame crude assertion as for above evaluation
	ASSERT(result->mean < 2);

	// clean up
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
}

}  // namespace my_shogun
