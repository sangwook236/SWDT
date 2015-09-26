//#include "stdafx.h"
#include <shogun/lib/config.h>
#include <shogun/base/init.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <cmath>


#if !defined(HAVE_EIGEN3) || !defined(HAVE_NLOPT)
#error this example requires Eigen3 & NLOPT libraries
#endif

namespace {
namespace local {

void build_matrices(const int32_t num_vectors, const int32_t dim_vectors, shogun::SGMatrix<float64_t> &test, shogun::SGMatrix<float64_t> &train, shogun::CRegressionLabels *labels)
{
	// Fill Matrices with random nonsense
	train[0] = -1;
	train[1] = -1;
	train[2] = -1;
	train[3] = 1;
	train[4] = 1;
	train[5] = 1;
	train[6] = -10;
	train[7] = -10;
	train[8] = -10;
	train[9] = 3;
	train[10] = 2;
	train[11] = 1;

	for (int32_t i = 0; i < num_vectors * dim_vectors; ++i)
		test[i] = i * std::sin(i) * 0.96;

	// create labels, two classes
	for (index_t i = 0; i < num_vectors; ++i)
	{
		if (i % 2 == 0) labels->set_label(i, 1);
		else labels->set_label(i, -1);
	}
}

shogun::CModelSelectionParameters * build_tree(shogun::CInferenceMethod *inf, shogun::CLikelihoodModel *lik, shogun::CKernel *kernel)
{
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();

	shogun::CModelSelectionParameters *c1 = new shogun::CModelSelectionParameters("inference_method", inf);
	root->append_child(c1);

	shogun::CModelSelectionParameters *c2 = new shogun::CModelSelectionParameters("scale");
	c1->append_child(c2);
	c2->build_values(0.01, 4.0, shogun::R_LINEAR);

	shogun::CModelSelectionParameters *c3 = new shogun::CModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3);

	shogun::CModelSelectionParameters *c4 = new shogun::CModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(0.01, 4.0, shogun::R_LINEAR);

	shogun::CModelSelectionParameters *c5 = new shogun::CModelSelectionParameters("kernel", kernel);
	c1->append_child(c5);

	shogun::CModelSelectionParameters *c6 = new shogun::CModelSelectionParameters("width");
	c5->append_child(c6);
	c6->build_values(0.01, 4.0, shogun::R_LINEAR);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/regression_gaussian_process_fitc.cpp
void regression_gaussian_process_fitc_example()
{
	const int32_t num_vectors = 4;
	const int32_t dim_vectors = 3;

	// create some data and labels
	shogun::SGMatrix<float64_t> matrix = shogun::SGMatrix<float64_t>(dim_vectors, num_vectors);

	shogun::SGMatrix<float64_t> matrix2 = shogun::SGMatrix<float64_t>(dim_vectors, num_vectors);

	shogun::CRegressionLabels *labels = new shogun::CRegressionLabels(num_vectors);

	local::build_matrices(num_vectors, dim_vectors, matrix2, matrix, labels);

	// create training features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>();
	features->set_feature_matrix(matrix);

	// create testing features
	shogun::CDenseFeatures<float64_t> *features2 = new shogun::CDenseFeatures<float64_t>();
	features2->set_feature_matrix(matrix2);

	SG_REF(labels);

	// Allocate our Kernel
	shogun::CGaussianKernel *test_kernel = new shogun::CGaussianKernel(10, 2);

	test_kernel->init(features, features);

	// Allocate our mean function
	shogun::CZeroMean *mean = new shogun::CZeroMean();

	// Allocate our likelihood function
	shogun::CGaussianLikelihood *lik = new shogun::CGaussianLikelihood();

	//SG_SPRINT("features2 bef inf rc = %d\n", features2->ref_count());

	// Allocate our inference method
	shogun::CFITCInferenceMethod *inf = new shogun::CFITCInferenceMethod(test_kernel, features, mean, labels, lik, features2);
	//SG_SPRINT("features2 aft inf rc = %d\n",features2->ref_count());
	SG_REF(inf);

	// Finally use these to allocate the Gaussian Process Object*/
	shogun::CGaussianProcessRegression *gp = new shogun::CGaussianProcessRegression(inf, features, labels);
	SG_REF(gp);

	// Build the parameter tree for model selection
	shogun::CModelSelectionParameters *root = local::build_tree(inf, lik, test_kernel);

	// Criterion for gradient search
	shogun::CGradientCriterion *crit = new shogun::CGradientCriterion();

	// This will evaluate our inference method for its derivatives
	shogun::CGradientEvaluation *grad = new shogun::CGradientEvaluation(gp, features, labels, crit);

	grad->set_function(inf);

	gp->print_modsel_params();
	root->print_tree();

	// handles all of the above structures in memory
	shogun::CGradientModelSelection *grad_search = new shogun::CGradientModelSelection(root, grad);

	// set autolocking to false to get rid of warnings
	grad->set_autolock(false);

	// Search for best parameters
	shogun::CParameterCombination *best_combination = grad_search->select_model(true);

	// Output all the results and information
	if (best_combination)
	{
		SG_SPRINT("best parameter(s):\n");
		best_combination->print_tree();

		best_combination->apply_to_machine(gp);
	}

	shogun::CGradientResult *result = (shogun::CGradientResult *)grad->evaluate();

	if (result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_SERROR("Evaluation result not a GradientEvaluationResult!");

	result->print_result();

	shogun::SGVector<float64_t> alpha = inf->get_alpha();
	shogun::SGVector<float64_t> labe = labels->get_labels();
	shogun::SGVector<float64_t> diagonal = inf->get_diagonal_vector();
	shogun::SGMatrix<float64_t> cholesky = inf->get_cholesky();

	gp->set_return_type(shogun::CGaussianProcessRegression::GP_RETURN_COV);
	shogun::CRegressionLabels *covariance = gp->apply_regression(features);

	gp->set_return_type(shogun::CGaussianProcessRegression::GP_RETURN_MEANS);
	shogun::CRegressionLabels *predictions = gp->apply_regression();

	alpha.display_vector("Alpha Vector");
	labe.display_vector("Labels");
	diagonal.display_vector("sW Matrix");
	covariance->get_labels().display_vector("Predicted Variances");
	predictions->get_labels().display_vector("Mean Predictions");
	cholesky.display_matrix("Cholesky Matrix L");
	matrix.display_matrix("Training Features");
	matrix2.display_matrix("Testing Features");

	// free memory
	SG_UNREF(predictions);
	SG_UNREF(covariance);
	SG_UNREF(labels);
	SG_UNREF(inf);
	SG_UNREF(gp);
	SG_UNREF(grad_search);
	SG_UNREF(best_combination);
	SG_UNREF(result);
}

}  // namespace my_shogun
