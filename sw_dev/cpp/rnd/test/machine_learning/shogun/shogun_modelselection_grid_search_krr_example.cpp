//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Labels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ParameterCombination.h>


namespace {
namespace local {

shogun::CModelSelectionParameters * create_param_tree()
{
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();

	shogun::CModelSelectionParameters *tau = new shogun::CModelSelectionParameters("tau");
	root->append_child(tau);
	tau->build_values(-1.0, 1.0, shogun::R_EXP);

	shogun::CGaussianKernel *gaussian_kernel = new shogun::CGaussianKernel();

	// print all parameter available for modelselection
	// Don't worry if yours is not included, simply write to the mailing list
	gaussian_kernel->print_modsel_params();

	shogun::CModelSelectionParameters *param_gaussian_kernel = new shogun::CModelSelectionParameters("kernel", gaussian_kernel);
	shogun::CModelSelectionParameters *gaussian_kernel_width = new shogun::CModelSelectionParameters("width");
	gaussian_kernel_width->build_values(5.0, 8.0, shogun::R_EXP, 1.0, 2.0);
	param_gaussian_kernel->append_child(gaussian_kernel_width);
	root->append_child(param_gaussian_kernel);

	shogun::CPolyKernel *poly_kernel = new shogun::CPolyKernel();

	// print all parameter available for modelselection
	// Dont worry if yours is not included, simply write to the mailing list
	poly_kernel->print_modsel_params();

	shogun::CModelSelectionParameters *param_poly_kernel = new shogun::CModelSelectionParameters("kernel", poly_kernel);
	root->append_child(param_poly_kernel);

	shogun::CModelSelectionParameters *param_poly_kernel_degree = new shogun::CModelSelectionParameters("degree");
	param_poly_kernel_degree->build_values(2, 3, shogun::R_LINEAR);
	param_poly_kernel->append_child(param_poly_kernel_degree);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/modelselection_grid_search_krr.cpp
void modelselection_grid_search_krr_example()
{
	// data matrix dimensions
	const index_t num_vectors = 30;
	const index_t num_features = 1;

	// training label data
	shogun::SGVector<float64_t> lab(num_vectors);

	// fill data matrix and labels
	shogun::SGMatrix<float64_t> train_dat(num_features, num_vectors);
	shogun::CMath::range_fill_vector(train_dat.matrix, num_vectors);
	for (index_t i = 0; i < num_vectors; ++i)
	{
		// labels are linear plus noise
		lab.vector[i] = i + shogun::CMath::normal_random(0, 1.0);
	}

	// training features
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	// training labels
	shogun::CLabels *labels = new shogun::CLabels(lab);

	// kernel ridge regression, only set labels for now, rest does not matter
	shogun::CKernelRidgeRegression *krr = new shogun::CKernelRidgeRegression(0, NULL, labels);

	// evaluation criterion
	shogun::CMeanSquaredError *eval_crit = new shogun::CMeanSquaredError();

	// splitting strategy
	const index_t n_folds = 5;
	shogun::CCrossValidationSplitting *splitting = new shogun::CCrossValidationSplitting(labels, n_folds);

	// cross validation instance, 10 runs, 95% confidence interval
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(krr, features, labels, splitting, eval_crit);
	cross->set_num_runs(3);
	cross->set_conf_int_alpha(0.05);

	// print all parameter available for modelselection
	// Don't worry if yours is not included, simply write to the mailing list
	krr->print_modsel_params();

	// model parameter selection, deletion is handled by modsel class (SG_UNREF)
	shogun::CModelSelectionParameters *param_tree = local::create_param_tree();
	param_tree->print_tree();

	// handles all of the above structures in memory
	shogun::CGridSearchModelSelection *grid_search = new shogun::CGridSearchModelSelection(param_tree, cross);

	// print current combination
	const bool print_state = true;
	shogun::CParameterCombination *best_combination = grid_search->select_model(print_state);
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(krr);

	// larger number of runs to have tighter confidence intervals
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.01);
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	// clean up
	SG_UNREF(features);
	SG_UNREF(best_combination);
	SG_UNREF(result);
	SG_UNREF(grid_search);
}

}  // namespace my_shogun
