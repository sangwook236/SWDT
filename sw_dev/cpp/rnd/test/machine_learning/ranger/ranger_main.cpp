#include "../ranger_lib/globals.h"
//#include "../ranger_lib/ArgumentHandler.h"
#include "../ranger_lib/ForestClassification.h"
#include "../ranger_lib/ForestRegression.h"
#include "../ranger_lib/ForestSurvival.h"
#include "../ranger_lib/ForestProbability.h"
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>


//#define _TRAINING_PHASE 1


namespace {
namespace local {

// REF [file] >> ${RANGER_HOME}/cpp_version/src/main.cpp
void iris_classification_example()
{
	// A training dataset in a file should contain one header line with variable names and one line with variable values per sample.
	//	Variable names must not contain any whitespace, comma, or semicolon.
	//	Values can be seperated by whitespace, comma, or semicolon but can not be mixed in one file.
#if _TRAINING_PHASE
	const std::string input_filename("./data/machine_learning/ranger/iris_training_modified.csv");
#else
	const std::string input_filename("./data/machine_learning/ranger/iris_test_modified.csv");
#endif
	// Filename to load forest and predict with new data.
	//	The new data is expected in the exact same shape as the training data.
	//	If the outcome of your new dataset is unknown, add a dummy column.
#if _TRAINING_PHASE
	const std::string load_forest_filename;  // prediction_mode = false if load_forest_filename is empty.
#else
	const std::string load_forest_filename("./data/machine_learning/ranger/iris_ranger_trained.forest");  // prediction_mode = true if load_forest_filename is not empty.
#endif

	const bool verbose = false;

	// Verbose output to logfile if non-verbose mode.
	std::unique_ptr<std::ostream> verbose_out;
#if 1
	verbose_out.reset(&std::cout);
#else
	verbose_out.reset(new std::ofstream("./data/machine_learning/ranger/iris.log"));
	if (!verbose_out->good())
		throw std::runtime_error("Could not write to logfile.");
#endif

	// Number of independent variables: sepal_length, sepal_width, petal_length, petal_width.
	const size_t num_independent_variables = 4;

	// Variable initialization.
	//	REF [file] >> ${RANGER_HOME}/cpp_version/src/utility/ArgumentHandler.cpp.
	//	REF [file] >> ${RANGER_HOME}/src/globals.h.

	const TreeType tree_type = TreeType::TREE_PROBABILITY;  // TREE_CLASSIFICATION, TREE_REGRESSION, TREE_SURVIVAL, TREE_PROBABILITY.
	const bool probability = false;  // Probability estimation (for classification forests only).
	const std::string dependent_variable_name("species");  // Name of dependent variable. For survival trees this is the time variable.
	const std::string status_variable_name;  // Name of status variable (for survival forests only). Coding is 1 for event and 0 for censored.
	const uint num_trees = 500;  // Number of trees.
	// Number of variables to possibly split at in each node.
	//	Default: sqrt(num_independent_variables) for classification and survival, num_independent_variables / 3 for regression.
	const uint mtry = std::floor(std::sqrt(num_independent_variables) + 0.5);
	// Minimal node size.
	//	For classification and regression growing is stopped if a node reaches a size smaller than N.
	//	For survival growing is stopped if one child would reach a size smaller than N.
	//	This means nodes with size smaller N can occur for classification and regression.
	//	Default: 1 for classification, 5 for regression, 3 for survival, and 10 for probability.
	const uint min_node_size = 1;
	// Comma separated list of names of (unordered) categorical variables.
	//	Categorical variables must contain only positive integer values.
	//std::vector<std::string> unordered_variable_names({ "sepal_length", "sepal_width", "petal_length", "petal_width" });  // Not categorical variable.
	std::vector<std::string> unordered_variable_names;

	// Return a matrix with individual predictions for each tree instead of aggregated predictions for all trees (classification and regression only).
	const bool predict_all = false;
	// Type of prediction.
	//	If TYPE = 1, predicted classes or values.
	//	If TYPE = 2, terminal node IDs per tree for new observations.
	const PredictionType prediction_type = PredictionType::RESPONSE;  // RESPONSE, TERMINALNODES.
	// Importance mode.
	//	If TYPE = 0, none.
	//	If TYPE = 1, node impurity: Gini for classification, variance for regression.
	//	If TYPE = 2, permutation importance, scaled by standard errors.
	//	If TYPE = 3, permutation importance, no scaling.
	// Node impurity variable importance (not supported for survival forests).
	const ImportanceMode importance_mode = ImportanceMode::IMP_NONE;  // IMP_NONE, IMP_GINI, IMP_PERM_BREIMAN, IMP_PERM_RAW, IMP_PERM_LIAW.
	const bool sample_with_replacement = true;  // Sample with replacement.
	// Fraction of observations to sample.
	//	Default is 1 for sampling with replacement and 0.632 for sampling without replacement.
	const double sample_fraction = 1.0;  // (0, 1].
	// Splitting rule.
	//	If RULE = 1, Gini for classification, variance for regression, logrank for survival.
	//	If RULE = 2, AUC for survival, not available for classification and regression.
	//	If RULE = 3, AUC(ignore ties) for survival, not available for classification and regression.
	//	If RULE = 4, MAXSTAT for survival and regression, not available for classification.
	//	If RULE = 5, ExtraTrees for all tree types.
	const SplitRule split_rule = SplitRule::LOGRANK;  // LOGRANK, AUC, AUC_IGNORE_TIES, MAXSTAT, EXTRATREES.
	const uint num_random_splits = 1;  // Number of random splits to consider for each splitting variable (ExtraTrees split_rule only).
	const double alpha = 0.5;  // Significance threshold to allow splitting (MAXSTAT split_rule only). [0, 1].
	const double minprop = 0.1;  // Lower quantile of covariate distribtuion to be considered for splitting (MAXSTAT split_rule only). [0, 0.5].
	const std::string case_weights_filename;  // Filename of case weights file.
	const bool holdout = false;  // Hold-out mode. Hold-out all samples with case weight 0 and use these for variable importance and prediction error.
	const std::string split_select_weights_file;  // Filename of split select weights file.
	std::vector<std::string> always_split_variable_names;  // Comma separated list of variable names to be always considered for splitting.

	const uint num_threads = 4;  // Number of parallel threads. Default: number of CPUs available.
	const uint seed = 0;  // Random seed.
	const std::string output_prefix("./data/machine_learning/ranger/iris_ranger_trained");  // Prefix for output files.

	// Memory mode.
	//	If MODE = 0, double.
	//	If MODE = 1, float.
	//	If MODE = 2, char.
	//	Default: 0.
	const MemoryMode memory_mode = MemoryMode::MEM_DOUBLE;  // MEM_DOUBLE, MEM_FLOAT, MEM_CHAR.
	const bool memory_saving_splitting = false;  // Use memory saving (but slower) splitting mode.

	// Create a random forest.
	std::unique_ptr<Forest> forest;
	switch (tree_type)
	{
	case TreeType::TREE_CLASSIFICATION:
		forest.reset(new ForestClassification);
		break;
	case TreeType::TREE_REGRESSION:
		forest.reset(new ForestRegression);
		break;
	case TreeType::TREE_SURVIVAL:
		forest.reset(new ForestSurvival);
		break;
	case TreeType::TREE_PROBABILITY:
		forest.reset(new ForestProbability);
		break;
	}
	if (!forest)
	{
		std::cerr << "ERROR: Invalid tree type." << std::endl;
		return;
	}

	// Initialize.
	forest->initCpp(
		dependent_variable_name, memory_mode, input_filename, mtry,
		output_prefix, num_trees, verbose_out.get(), seed, num_threads,
		load_forest_filename, importance_mode, min_node_size, split_select_weights_file,
		always_split_variable_names, status_variable_name, sample_with_replacement, unordered_variable_names,
		memory_saving_splitting, split_rule, case_weights_filename, predict_all, sample_fraction,
		alpha, minprop, holdout, prediction_type,
		num_random_splits
	);

	// Grow (if prediction_mode != true) or predict (if prediction_mode == true).
	std::cout << "Start growing(training) or prediction(testing)..." << std::endl;
	{
		boost::timer::auto_cpu_timer timer;
		forest->run(verbose);
	}
	std::cout << "End growing(training) or prediction(testing)..." << std::endl;

#if _TRAINING_PHASE
	// Save.
	forest->saveToFile();  // Save forest to file <output_prefix>.forest.
#endif

	// Output the result.
	forest->writeOutput();

	// Clean-up.
	verbose_out.release();
	forest.release();
}

}  // namespace local
}  // unnamed namespace

namespace my_ranger {

}  // namespace my_ranger

int ranger_main(int argc, char *argv[])
{
	local::iris_classification_example();

	return 0;
}
