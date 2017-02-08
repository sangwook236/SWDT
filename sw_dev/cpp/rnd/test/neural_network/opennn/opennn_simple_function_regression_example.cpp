//#include "stdafx.h"
#include <opennn/opennn.h>
#include <iostream>
#include <sstream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opennn {

// REF [file] >> ${OPENNN_HOME}/examples/simple_function_regression/main.cpp
void simple_function_regression_example()
{
	std::cout << "OpenNN. Simple Function Regression Application." << std::endl;

	//std::srand((unsigned)std:time(NULL));

	// Data set object
	OpenNN::DataSet data_set;
	data_set.set_data_file_name("./data/neural_network/opennn/simple_function_regression.dat");

	data_set.load_data();

	OpenNN::Variables* variables_pointer = data_set.get_variables_pointer();

	variables_pointer->set_use(0, OpenNN::Variables::Input);
	variables_pointer->set_use(1, OpenNN::Variables::Target);

	variables_pointer->set_name(0, "x");
	variables_pointer->set_name(1, "y");

	OpenNN::Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
	OpenNN::Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

	OpenNN::Instances* instances_pointer = data_set.get_instances_pointer();

	instances_pointer->set_training();

	OpenNN::Vector<OpenNN::Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
	OpenNN::Vector<OpenNN::Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

	// Neural network.
	OpenNN::NeuralNetwork neural_network(1, 15, 1);

	OpenNN::Inputs* inputs_pointer = neural_network.get_inputs_pointer();
	inputs_pointer->set_information(inputs_information);

	OpenNN::Outputs* outputs_pointer = neural_network.get_outputs_pointer();
	outputs_pointer->set_information(targets_information);

	neural_network.construct_scaling_layer();
	OpenNN::ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
	scaling_layer_pointer->set_statistics(inputs_statistics);
	scaling_layer_pointer->set_scaling_method(OpenNN::ScalingLayer::NoScaling);

	neural_network.construct_unscaling_layer();
	OpenNN::UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
	unscaling_layer_pointer->set_statistics(targets_statistics);
	unscaling_layer_pointer->set_unscaling_method(OpenNN::UnscalingLayer::NoUnscaling);

	// Performance functional object.
	OpenNN::PerformanceFunctional performance_functional(&neural_network, &data_set);

	// Training strategy.
	OpenNN::TrainingStrategy training_strategy(&performance_functional);

	OpenNN::QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

	quasi_Newton_method_pointer->set_minimum_performance_increase(1.0e-3);

	OpenNN::TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

	// Testing analysis object.
	instances_pointer->set_testing();

	OpenNN::TestingAnalysis testing_analysis(&neural_network, &data_set);

	OpenNN::TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();

	// Save results.
	data_set.save("./data/neural_network/opennn/simple_function_regression/data_set.xml");

	neural_network.save("./data/neural_network/opennn/simple_function_regression/neural_network.xml");
	neural_network.save_expression("./data/neural_network/opennn/simple_function_regression/expression.txt");

	performance_functional.save("./data/neural_network/opennn/simple_function_regression/performance_functional.xml");

	training_strategy.save("./data/neural_network/opennn/simple_function_regression/training_strategy.xml");
	training_strategy_results.save("./data/neural_network/opennn/simple_function_regression/training_strategy_results.dat");

	linear_regression_results.save("./data/neural_network/opennn/simple_function_regression/linear_regression_analysis_results.dat");
}

}  // namespace my_opennn
