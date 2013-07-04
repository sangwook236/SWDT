//#include "stdafx.h"
#include <shogun/lib/config.h>
#include <shogun/base/init.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/kernel/ProductKernel.h>
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

shogun::CModelSelectionParameters * build_tree(shogun::CInferenceMethod *inf, shogun::CLikelihoodModel *lik, shogun::CProductKernel *kernel)
{		
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();

	shogun::CModelSelectionParameters *c1 = new shogun::CModelSelectionParameters("inference_method", inf);
	root->append_child(c1);

	shogun::CModelSelectionParameters *c2 = new shogun::CModelSelectionParameters("scale");
	c1->append_child(c2);
	c2->build_values(0.99, 1.01, R_LINEAR);

	shogun::CModelSelectionParameters *c3 =  new shogun::CModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3); 

	shogun::CModelSelectionParameters *c4 = new shogun::CModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(1.0, 4.0, R_LINEAR);

	shogun::CModelSelectionParameters *c5 = new shogun::CModelSelectionParameters("kernel", kernel);
	c1->append_child(c5);

	shogun::CList *list = kernel->get_list();
	shogun::CModelSelectionParameters *cc1 = new CModelSelectionParameters("kernel_list", list);
	c5->append_child(cc1);

	shogun::CListElement *first = NULL;
	shogun::CSGObject *k = list->get_first_element(first);
	SG_UNREF(k);
	SG_REF(first);

	shogun::CModelSelectionParameters *cc2 = new shogun::CModelSelectionParameters("first", first);
	cc1->append_child(cc2);

	shogun::CKernel *sub_kernel1 = kernel->get_kernel(0);
	shogun::CModelSelectionParameters *cc3 = new shogun::CModelSelectionParameters("data", sub_kernel1);
	cc2->append_child(cc3);
	SG_UNREF(sub_kernel1);

	shogun::CListElement *second = first;
	k = list->get_next_element(second);
	SG_UNREF(k);
	SG_REF(second);

	shogun::CModelSelectionParameters *cc4 = new shogun::CModelSelectionParameters("next", second);
	cc2->append_child(cc4);

	shogun::CKernel *sub_kernel2 = kernel->get_kernel(1);
	shogun::CModelSelectionParameters *cc5 = new shogun::CModelSelectionParameters("data", sub_kernel2);
	cc4->append_child(cc5);
	SG_UNREF(sub_kernel2);

	shogun::CListElement *third = second;
	k = list->get_next_element(third);
	SG_UNREF(k);
	SG_REF(third);

	shogun::CModelSelectionParameters *cc6 = new CModelSelectionParameters("next", third);
	cc4->append_child(cc6);

	shogun::CKernel *sub_kernel3 = kernel->get_kernel(2);
	shogun::CModelSelectionParameters *cc7 = new shogun::CModelSelectionParameters("data", sub_kernel3);
	cc6->append_child(cc7);
	SG_UNREF(sub_kernel3);

	shogun::CModelSelectionParameters *c6 = new shogun::CModelSelectionParameters("width");
	cc3->append_child(c6);
	c6->build_values(1.0, 4.0, R_LINEAR);

	shogun::CModelSelectionParameters *c66 = new shogun::CModelSelectionParameters("combined_kernel_weight");
	cc3->append_child(c66);
	c66->build_values(0.001, 1.0, R_LINEAR);

	shogun::CModelSelectionParameters *c7 = new shogun::CModelSelectionParameters("width");
	cc5->append_child(c7);
	c7->build_values(1.0, 4.0, R_LINEAR);

	shogun::CModelSelectionParameters* c77 = new shogun::CModelSelectionParameters("combined_kernel_weight");
	cc5->append_child(c77);
	c77->build_values(0.001, 1.0, R_LINEAR);

	shogun::CModelSelectionParameters *c8 = new shogun::CModelSelectionParameters("width");
	cc7->append_child(c8);
	c8->build_values(1.0, 4.0, R_LINEAR);

	shogun::CModelSelectionParameters *c88 = new shogun::CModelSelectionParameters("combined_kernel_weight");
	cc7->append_child(c88);
	c88->build_values(0.001, 1.0, R_LINEAR);

	SG_UNREF(list);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/regression_gaussian_process_product.cpp
void regression_gaussian_process_product_example()
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

	shogun::CCombinedFeatures *comb_features = new shogun::CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	shogun::CProductKernel *test_kernel = new shogun::CProductKernel();
	shogun::CGaussianKernel *sub_kernel1 = new shogun::CGaussianKernel(10, 2);
	shogun::CGaussianKernel *sub_kernel2 = new shogun::CGaussianKernel(10, 2);
	shogun::CGaussianKernel *sub_kernel3 = new shogun::CGaussianKernel(10, 2);

	test_kernel->append_kernel(sub_kernel1);
	test_kernel->append_kernel(sub_kernel2);
	test_kernel->append_kernel(sub_kernel3);

	SG_REF(comb_features);
	SG_REF(labels);

	// Allocate our Mean Function
	shogun::CZeroMean *mean = new shogun::CZeroMean();

	// Allocate our Likelihood Model
	shogun::CGaussianLikelihood *lik = new shogun::CGaussianLikelihood();

	// Allocate our inference method
	shogun::CExactInferenceMethod *inf = new shogun::CExactInferenceMethod(test_kernel, comb_features, mean, labels, lik);
	SG_REF(inf);

	// Finally use these to allocate the Gaussian Process Object
	shogun::CGaussianProcessRegression *gp = new shogun::CGaussianProcessRegression(inf, comb_features, labels);
	SG_REF(gp);

	shogun::CModelSelectionParameters *root = local::build_tree(inf, lik, test_kernel);

	// Criterion for gradient search
	shogun::CGradientCriterion *crit = new shogun::CGradientCriterion();

	// This will evaluate our inference method for its derivatives
	shogun::CGradientEvaluation *grad = new shogun::CGradientEvaluation(gp, comb_features, labels, crit);

	grad->set_function(inf);

	gp->print_modsel_params();
	root->print_tree();

	// handles all of the above structures in memory
	shogun::CGradientModelSelection *grad_search = new shogun::CGradientModelSelection(root, grad);

	// set autolocking to false to get rid of warnings
	grad->set_autolock(false);

	// Search for best parameters
	CParameterCombination *best_combination = grad_search->select_model(true);

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
	shogun::CRegressionLabels *covariance = gp->apply_regression(comb_features);

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
	SG_UNREF(comb_features);
	SG_UNREF(inf);
	SG_UNREF(gp);
	SG_UNREF(grad_search);
	SG_UNREF(best_combination);
	SG_UNREF(result);
}

}  // namespace my_shogun
