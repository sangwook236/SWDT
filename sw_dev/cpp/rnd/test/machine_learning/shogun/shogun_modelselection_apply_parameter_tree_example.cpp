//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>


namespace {
namespace local {

using namespace shogun;

shogun::CModelSelectionParameters * create_param_tree()
{
	shogun::CModelSelectionParameters *root = new shogun::CModelSelectionParameters();

	shogun::CModelSelectionParameters *c = new shogun::CModelSelectionParameters("C1");
	root->append_child(c);
	c->build_values(1.0, 2.0, shogun::R_EXP);

	shogun::CGaussianKernel *gaussian_kernel = new shogun::CGaussianKernel();

	// print all parameter available for modelselection.
	// Don't worry if yours is not included, simply write to the mailing list.
	gaussian_kernel->print_modsel_params();

	shogun::CModelSelectionParameters *param_gaussian_kernel = new shogun::CModelSelectionParameters("kernel", gaussian_kernel);

	root->append_child(param_gaussian_kernel);

	shogun::CModelSelectionParameters *param_gaussian_kernel_width = new shogun::CModelSelectionParameters("width");
	param_gaussian_kernel_width->build_values(1.0, 2.0, shogun::R_EXP);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	return root;
}

void apply_parameter_tree(shogun::CDynamicObjectArray *combinations)
{
	// create some data
	shogun::SGMatrix<float64_t> matrix(2, 3);
	for (index_t i = 0; i < 6; ++i)
		matrix.matrix[i] = i;

	// create three 2-dimensional vectors to avoid deleting these, REF now and UNREF when finished
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(matrix);
	SG_REF(features);

	// create three labels, will be handed to svm and automaticall deleted
	shogun::CBinaryLabels *labels = new shogun::CBinaryLabels(3);
	SG_REF(labels);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);

	// create libsvm with C=10 and train
	shogun::CLibSVM *svm = new shogun::CLibSVM();
	SG_REF(svm);
	svm->set_labels(labels);

	for (index_t i = 0; i < combinations->get_num_elements(); ++i)
	{
		SG_SPRINT("applying:\n");
		shogun::CParameterCombination *current_combination = (shogun::CParameterCombination *)combinations->get_element(i);
		current_combination->print_tree();
		shogun::Parameter *current_parameters = svm->m_parameters;
		current_combination->apply_to_modsel_parameter(current_parameters);
		SG_UNREF(current_combination);

		// get kernel to set features, get_kernel SG_REF's the kernel
		shogun::CKernel *kernel = svm->get_kernel();
		kernel->init(features, features);

		svm->train();

		// classify on training examples
		for (index_t i = 0; i < 3; ++i)
			SG_SPRINT("output[%d ]= %f\n", i, svm->apply_one(i));

		// unset features and SG_UNREF kernel
		kernel->cleanup();
		SG_UNREF(kernel);

		SG_SPRINT("----------------\n\n");
	}

	// free up memory
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(svm);
}

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/modelselection_apply_parameter_tree.cpp
void modelselection_apply_parameter_tree_example()
{
	// create example tree
	shogun::CModelSelectionParameters *tree = local::create_param_tree();
	tree->print_tree();
	SG_SPRINT("----------------------------------\n");

	// build combinations of parameter trees
	shogun::CDynamicObjectArray *combinations = tree->get_combinations();
	local::apply_parameter_tree(combinations);

	// print and directly delete them all
	for (index_t i = 0; i < combinations->get_num_elements(); ++i)
	{
		shogun::CParameterCombination *combination = (shogun::CParameterCombination *)combinations->get_element(i);
		SG_UNREF(combination);
	}

	SG_UNREF(combinations);

	// delete example tree (after processing of combinations because CSGObject (namely the kernel) of the tree is SG_UNREF'ed (and not REF'ed anywhere else)
	SG_UNREF(tree);
}

}  // namespace my_shogun
