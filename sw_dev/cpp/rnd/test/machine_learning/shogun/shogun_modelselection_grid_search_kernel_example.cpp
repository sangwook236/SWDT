//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>


namespace {
namespace local {

shogun::CModelSelectionParameters * create_param_tree()
{
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();

	shogun::CModelSelectionParameters *c1 = new shogun::CModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(-1.0, 1.0, shogun::R_EXP);

	shogun::CModelSelectionParameters *c2 = new shogun::CModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-1.0, 1.0, shogun::R_EXP);

	shogun::CGaussianKernel *gaussian_kernel = new shogun::CGaussianKernel();

	// print all parameter available for modelselection
	// Don't worry if yours is not included, simply write to the mailing list
	gaussian_kernel->print_modsel_params();

	shogun::CModelSelectionParameters *param_gaussian_kernel = new shogun::CModelSelectionParameters("kernel", gaussian_kernel);
	shogun::CModelSelectionParameters *gaussian_kernel_width = new shogun::CModelSelectionParameters("width");
	gaussian_kernel_width->build_values(-1.0, 1.0, shogun::R_EXP, 1.0, 2.0);
	param_gaussian_kernel->append_child(gaussian_kernel_width);
	root->append_child(param_gaussian_kernel);

	shogun::CPowerKernel *power_kernel = new shogun::CPowerKernel();

	// print all parameter available for modelselection
	// Don't worry if yours is not included, simply write to the mailing list
	power_kernel->print_modsel_params();

	shogun::CModelSelectionParameters *param_power_kernel= new shogun::CModelSelectionParameters("kernel", power_kernel);

	root->append_child(param_power_kernel);

	shogun::CModelSelectionParameters *param_power_kernel_degree = new shogun::CModelSelectionParameters("degree");
	param_power_kernel_degree->build_values(1.0, 2.0, shogun::R_LINEAR);
	param_power_kernel->append_child(param_power_kernel_degree);

	shogun::CMinkowskiMetric *m_metric = new shogun::CMinkowskiMetric(10);

	// print all parameter available for modelselection
	// Don/t worry if yours is not included, simply write to the mailing list
	m_metric->print_modsel_params();

	shogun::CModelSelectionParameters *param_power_kernel_metric1 = new shogun::CModelSelectionParameters("distance", m_metric);
	param_power_kernel->append_child(param_power_kernel_metric1);

	shogun::CModelSelectionParameters *param_power_kernel_metric1_k = new shogun::CModelSelectionParameters("k");
	param_power_kernel_metric1_k->build_values(1.0, 2.0, shogun::R_LINEAR);
	param_power_kernel_metric1->append_child(param_power_kernel_metric1_k);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/modelselection_grid_search_kernel.cpp
void modelselection_grid_search_kernel_example()
{
	const int32_t num_subsets = 3;
	const int32_t num_vectors = 20;
	const int32_t dim_vectors = 3;

	// create some data and labels
	shogun::SGMatrix<float64_t> matrix(dim_vectors, num_vectors);
	shogun::CBinaryLabels *labels = new shogun::CBinaryLabels(num_vectors);

	for (int32_t i = 0; i < num_vectors * dim_vectors; ++i)
		matrix.matrix[i] = shogun::CMath::randn_double();

	// create num_feautres 2-dimensional vectors
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(matrix);

	// create labels, two classes
	for (index_t i = 0; i < num_vectors; ++i)
		labels->set_label(i, i % 2 == 0 ? 1 : -1);

	// create svm
	shogun::CLibSVM *classifier = new shogun::CLibSVM();

	// splitting strategy
	shogun::CStratifiedCrossValidationSplitting *splitting_strategy = new shogun::CStratifiedCrossValidationSplitting(labels, num_subsets);

	// accuracy evaluation
	shogun::CContingencyTableEvaluation *evaluation_criterium = new shogun::CContingencyTableEvaluation(shogun::ACCURACY);

	// cross validation class for evaluation in model selection
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(classifier, features, labels, splitting_strategy, evaluation_criterium);
	cross->set_num_runs(1);
	// note that this automatically is not necessary since done automatically
	cross->set_autolock(true);

	// print all parameter available for modelselection
	// Don't worry if yours is not included, simply write to the mailing list
	classifier->print_modsel_params();

	// model parameter selection, deletion is handled by modsel class (SG_UNREF)
	shogun::CModelSelectionParameters *param_tree = local::create_param_tree();
	param_tree->print_tree();

	// handles all of the above structures in memory
	shogun::CGridSearchModelSelection *grid_search = new shogun::CGridSearchModelSelection(param_tree, cross);

	const bool print_state = true;
	shogun::CParameterCombination *best_combination = grid_search->select_model(print_state);
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	// larger number of runs to have tighter confidence intervals
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.01);
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	// now again but unlocked
	SG_UNREF(best_combination);
	cross->set_autolock(true);
	best_combination = grid_search->select_model(print_state);
	best_combination->apply_to_machine(classifier);
	SG_UNREF(result);
	result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result (unlocked): ");

	// clean up destroy result parameter
	SG_UNREF(result);
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);
}

}  // namespace my_shogun
