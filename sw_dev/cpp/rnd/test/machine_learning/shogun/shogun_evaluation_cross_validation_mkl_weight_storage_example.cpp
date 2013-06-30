//#include "stdafx.h"
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationPrintOutput.h>
#include <shogun/evaluation/CrossValidationMKLStorage.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Statistics.h>


namespace {
namespace local {

void gen_rand_data(shogun::SGVector<float64_t> lab, shogun::SGMatrix<float64_t> feat, const float64_t dist)
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

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/evaluation_cross_validation_mkl_weight_storage.cpp
void evaluation_cross_validation_mkl_weight_storage_example()
{
	// generate random data
	const index_t num = 10;
	const index_t dims = 2;
	const float64_t dist = 0.5;
	shogun::SGVector<float64_t> lab(num);
	shogun::SGMatrix<float64_t> feat(dims, num);
	local::gen_rand_data(lab, feat, dist);

	// create train labels
	shogun::CLabels *labels = new shogun::CBinaryLabels(lab);

	// create train features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>();
	features->set_feature_matrix(feat);
	SG_REF(features);

	// create combined features
	shogun::CCombinedFeatures *comb_features = new shogun::CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	SG_REF(comb_features);

	// create multiple gaussian kernels
	shogun::CCombinedKernel *kernel = new shogun::CCombinedKernel();
	kernel->append_kernel(new shogun::CGaussianKernel(10, 0.1));
	kernel->append_kernel(new shogun::CGaussianKernel(10, 1));
	kernel->append_kernel(new shogun::CGaussianKernel(10, 2));
	kernel->init(comb_features, comb_features);
	SG_REF(kernel);

	// create mkl using libsvm, due to a mem-bug, interleaved is not possible
	shogun::CMKLClassification *svm = new shogun::CMKLClassification(new shogun::CLibSVM());
	svm->set_interleaved_optimization_enabled(false);
	svm->set_kernel(kernel);
	SG_REF(svm);

	// create cross-validation instance
	const index_t num_folds = 3;
	shogun::CSplittingStrategy *split = new shogun::CStratifiedCrossValidationSplitting(labels, num_folds);
	shogun::CEvaluation *eval = new shogun::CContingencyTableEvaluation(ACCURACY);
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(svm, comb_features, labels, split, eval, false);

	// add print output listener and mkl storage listener
	cross->add_cross_validation_output(new shogun::CCrossValidationPrintOutput());
	shogun::CCrossValidationMKLStorage *mkl_storage = new shogun::CCrossValidationMKLStorage();
	cross->add_cross_validation_output(mkl_storage);

	// perform cross-validation, this will print loads of information (caused by the CCrossValidationPrintOutput instance attached to it)
	shogun::CEvaluationResult *result = cross->evaluate();

	// print mkl weights
	shogun::SGMatrix<float64_t> weights = mkl_storage->get_mkl_weights();
	weights.display_matrix("mkl weights");

	// print mean and variance of each kernel weight.
	// These could for example been used to compute confidence intervals
	shogun::CStatistics::matrix_mean(weights, false).display_vector("mean per kernel");
	shogun::CStatistics::matrix_variance(weights, false).display_vector("variance per kernel");
	shogun::CStatistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	SG_UNREF(result);

	// again for two runs
	cross->set_num_runs(2);
	result = cross->evaluate();

	// print mkl weights
	weights = mkl_storage->get_mkl_weights();
	weights.display_matrix("mkl weights");

	// print mean and variance of each kernel weight.
	// These could for example been used to compute confidence intervals
	shogun::CStatistics::matrix_mean(weights, false).display_vector("mean per kernel");
	shogun::CStatistics::matrix_variance(weights, false).display_vector("variance per kernel");
	shogun::CStatistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	// clean up
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(comb_features);
	SG_UNREF(svm);
}

}  // namespace my_shogun
