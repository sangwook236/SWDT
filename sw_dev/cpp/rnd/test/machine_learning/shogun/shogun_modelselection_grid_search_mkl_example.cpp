//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/features/Labels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/io/SGIO.h>


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

	shogun::CCombinedKernel *kernel1 = new shogun::CCombinedKernel();
	kernel1->append_kernel(new shogun::CGaussianKernel(10, 2));
	kernel1->append_kernel(new shogun::CGaussianKernel(10, 3));
	kernel1->append_kernel(new shogun::CGaussianKernel(10, 4));

	shogun::CModelSelectionParameters *param_kernel1 = new shogun::CModelSelectionParameters("kernel", kernel1);
	root->append_child(param_kernel1);

	shogun::CCombinedKernel *kernel2 = new shogun::CCombinedKernel();
	kernel2->append_kernel(new shogun::CGaussianKernel(10, 20));
	kernel2->append_kernel(new shogun::CGaussianKernel(10, 30));
	kernel2->append_kernel(new shogun::CGaussianKernel(10, 40));

	shogun::CModelSelectionParameters *param_kernel2 = new shogun::CModelSelectionParameters("kernel", kernel2);
	root->append_child(param_kernel2);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/modelselection_grid_search_mkl.cpp
void modelselection_grid_search_mkl_example()
{
	const int32_t num_subsets = 3;
	const int32_t num_vectors = 20;
	const int32_t dim_vectors = 3;

	// create some data and labels
	float64_t *matrix = SG_MALLOC(float64_t, num_vectors * dim_vectors);
	//--S [] 2013/07/04: Sang-Wook Lee
	//shogun::CLabels *labels = new shogun::CLabels(num_vectors);
	shogun::CBinaryLabels *labels = new shogun::CBinaryLabels(num_vectors);
	//--E [] 2013/07/04: Sang-Wook Lee
	for (int32_t i = 0; i < num_vectors * dim_vectors; ++i)
		matrix[i] = shogun::CMath::randn_double();

	// create num_feautres 2-dimensional vectors
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>();
	//--S [] 2013/07/04: Sang-Wook Lee
	//features->set_feature_matrix(matrix, dim_vectors, num_vectors);
	features->set_feature_matrix(matrix);
	//--E [] 2013/07/04: Sang-Wook Lee

	// create combined features
	shogun::CCombinedFeatures *comb_features = new shogun::CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	// create labels, two classes
	for (index_t i = 0; i < num_vectors; ++i)
		labels->set_label(i, i % 2 == 0 ? 1 : -1);

	// works
	// create svm
	//shogun::CMKLClassification *classifier = new shogun::CMKLClassification(new shogun::CLibSVM());
	//classifier->set_interleaved_optimization_enabled(false);

	// create svm
	shogun::CMKLClassification *classifier = new shogun::CMKLClassification();

	// both fail:
	//classifier->set_interleaved_optimization_enabled(false);
	classifier->set_interleaved_optimization_enabled(true);

	// splitting strategy
	shogun::CStratifiedCrossValidationSplitting *splitting_strategy = new shogun::CStratifiedCrossValidationSplitting(labels, num_subsets);

	// accuracy evaluation
	shogun::CContingencyTableEvaluation *evaluation_criterium = new shogun::CContingencyTableEvaluation(shogun::ACCURACY);

	// cross validation class for evaluation in model selection
	shogun::CCrossValidation *cross = new shogun::CCrossValidation(classifier, comb_features, labels, splitting_strategy, evaluation_criterium);
	cross->set_num_runs(1);

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
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	// larger number of runs to have tighter confidence intervals
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.01);
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
        //--S [] 2013/07/04: Sang-Wook Lee
		//SG_ERROR("Evaluation result is not of type CCrossValidationResult!");
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");
        //--E [] 2013/07/04: Sang-Wook Lee

	SG_SPRINT("result: ");
	result->print_result();

	// clean up destroy result parameter
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);
}

}  // namespace my_shogun
