//#include "stdafx.h"
#include <opennn/opennn.h>
#include <iostream>
#include <sstream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opennn {

// [ref] ${FANN_HOME}/examples/simple_pattern_recognition/simple_function_regression_application.cpp
void simple_function_regression_example()
{
	std::cout << "OpenNN. Simple Function Regression Application." << std::endl;

	// Data set object
	OpenNN::DataSet data_set;
	data_set.load_data("./data/neural_network/opennn/simple_function_regression.dat");

	OpenNN::VariablesInformation* variables_information_pointer = data_set.get_variables_information_pointer();

	variables_information_pointer->set_name(0, "x");   
	variables_information_pointer->set_name(1, "y");   

	OpenNN::Vector<OpenNN::Vector<std::string> > inputs_targets_information = variables_information_pointer->arrange_inputs_targets_information();

	OpenNN::InstancesInformation *instances_information_pointer = data_set.get_instances_information_pointer();
	instances_information_pointer->split_random_indices(0.75, 0.1, 0.25);

	const OpenNN::Vector<OpenNN::Vector<double> > inputs_targets_minimum_maximum = data_set.scale_inputs_targets_minimum_maximum();

	// Neural network
	OpenNN::NeuralNetwork neural_network(1, 3, 1);
	neural_network.set_inputs_outputs_information(inputs_targets_information);
	neural_network.set_inputs_outputs_minimums_maximums(inputs_targets_minimum_maximum);

	// Performance functional object
	OpenNN::PerformanceFunctional performance_functional(&neural_network, &data_set);

	// Training strategy
	OpenNN::TrainingStrategy training_strategy(&performance_functional);
	training_strategy.perform_training();

	neural_network.set_inputs_scaling_outputs_unscaling_methods("MeanStandardDeviation");

	// Testing analysis object
	OpenNN::TestingAnalysis testing_analysis(&neural_network, &data_set);
	OpenNN::FunctionRegressionTesting::LinearRegressionAnalysisResults linear_regression_analysis_results = testing_analysis.get_function_regression_testing_pointer()->perform_linear_regression_analysis();

	std::cout << "Linear regression parameters:" << std::endl
		<< "Intercept: " << linear_regression_analysis_results.linear_regression_parameters[0][0] << std::endl
		<< "Slope: " << linear_regression_analysis_results.linear_regression_parameters[0][1] << std::endl;

	// Save results
	data_set.save("./data/neural_network/opennn/simple_function_regression/data_set.xml");

	neural_network.save("./data/neural_network/opennn/simple_function_regression/neural_network.xml");
	neural_network.save_expression("./data/neural_network/opennn/simple_function_regression/expression.txt");

	performance_functional.save("./data/neural_network/opennn/simple_function_regression/performance_functional.xml");

	training_strategy.save("./data/neural_network/opennn/simple_function_regression/training_strategy.xml");

	linear_regression_analysis_results.save("./data/neural_network/opennn/simple_function_regression/linear_regression_analysis_results.dat");
}

}  // namespace my_opennn
