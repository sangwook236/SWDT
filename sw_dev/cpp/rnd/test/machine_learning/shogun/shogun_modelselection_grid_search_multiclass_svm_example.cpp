//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

shogun::CModelSelectionParameters * build_param_tree(shogun::CKernel *kernel)
{
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();
	shogun::CModelSelectionParameters *c = new shogun::CModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(-1.0, 1.0, shogun::R_EXP);

	shogun::CModelSelectionParameters *params_kernel = new shogun::CModelSelectionParameters("kernel", kernel);
	root->append_child(params_kernel);
	shogun::CModelSelectionParameters *params_kernel_width = new shogun::CModelSelectionParameters("width");
	params_kernel_width->build_values(-1.0, 1.0, shogun::R_EXP);
	params_kernel->append_child(params_kernel_width);

	return root;
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/modelselection_grid_search_multiclass_svm.cpp
void modelselection_grid_search_multiclass_svm_example()
{
	// number of classes is dimension of data here to have some easy multiclass structure
	const unsigned int num_vectors = 50;
	const unsigned int dim_vectors = 3;
	// Heiko: increase number of classes and things will fail :(
	// Sergey: the special buggy case of 3 classes was hopefully fixed

	const float64_t distance = 5;

	// create data: some easy multiclass data
	shogun::SGMatrix<float64_t> feat = shogun::SGMatrix<float64_t>(dim_vectors, num_vectors);
	shogun::SGVector<float64_t> lab(num_vectors);
	for (index_t j = 0; j < feat.num_cols; ++j)
	{
		lab[j] = j % dim_vectors;

		for (index_t i = 0; i < feat.num_rows; ++i)
			feat(i, j) = shogun::CMath::randn_double();

		// make sure classes are (alomst) linearly seperable against each other
		feat(lab[j], j) += distance;
	}

	// shogun representation of above data
	shogun::CDenseFeatures<float64_t> *cfeatures = new shogun::CDenseFeatures<float64_t>(feat);
	shogun::CMulticlassLabels *clabels = new shogun::CMulticlassLabels(lab);

	const float64_t sigma = 2;
	shogun::CGaussianKernel *kernel = new shogun::CGaussianKernel(10, sigma);

	const float C = 10.;
	shogun::CMulticlassLibSVM *cmachine = new shogun::CMulticlassLibSVM(C, kernel, clabels);

	shogun::CMulticlassAccuracy *eval_crit = new shogun::CMulticlassAccuracy();

	// k-fold stratified x-validation
	const index_t k = 3;
	shogun::CStratifiedCrossValidationSplitting *splitting = new shogun::CStratifiedCrossValidationSplitting(clabels, k);

	shogun::CCrossValidation *cross = new shogun::CCrossValidation(cmachine, cfeatures, clabels, splitting, eval_crit);
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.05);

	// create peramters for model selection
	shogun::CModelSelectionParameters *root = local::build_param_tree(kernel);

	shogun::CGridSearchModelSelection *model_selection = new shogun::CGridSearchModelSelection(cross, root);
	const bool print_state = true;
	shogun::CParameterCombination *params = model_selection->select_model(print_state);
	SG_SPRINT("best combination\n");
	params->print_tree();

	// clean up memory
	SG_UNREF(model_selection);
	SG_UNREF(params);
}

}  // namespace my_shogun
