//#include "stdafx.h"
#include <opennn/opennn.h>
#include <iostream>
#include <sstream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opennn {

// [ref] ${FANN_HOME}/examples/simple_pattern_recognition/simple_pattern_recognition_application.cpp
void simple_pattern_recognition_example()
{
	std::cout << "OpenNN. Simple Pattern Recognition Application." << std::endl;

	// Data set object
	OpenNN::DataSet data_set;
	data_set.load_data("./data/neural_network/opennn/simple_pattern_recognition.dat");

	OpenNN::VariablesInformation *variables_information_pointer = data_set.get_variables_information_pointer();

	variables_information_pointer->set_name(0, "x1");   
	variables_information_pointer->set_name(1, "x2");   
	variables_information_pointer->set_name(2, "y");   

	OpenNN::InstancesInformation *instances_information_pointer = data_set.get_instances_information_pointer();

	instances_information_pointer->split_random_indices(0.75, 0.0, 0.25);

	OpenNN::Vector<OpenNN::Vector<std::string> > variables_information = variables_information_pointer->arrange_inputs_targets_information();

	const OpenNN::Vector<OpenNN::Vector<double> > variables_statistics = data_set.scale_inputs();

	// Neural network object
	OpenNN::NeuralNetwork neural_network(2, 2, 1);

	neural_network.set_inputs_outputs_information(variables_information);
	neural_network.set_inputs_outputs_statistics(variables_statistics);

	neural_network.set_scaling_layer_flag(false);

	// Performance functional
	OpenNN::PerformanceFunctional performance_functional(&neural_network, &data_set);

	// Training strategy
	OpenNN::TrainingStrategy training_strategy(&performance_functional);
	OpenNN::TrainingStrategy::Results training_stategy_results = training_strategy.perform_training();

	neural_network.set_scaling_layer_flag(true);

	// Testing analysis
	OpenNN::TestingAnalysis testing_analysis(&neural_network, &data_set);
	testing_analysis.construct_pattern_recognition_testing();

	OpenNN::PatternRecognitionTesting *pattern_recognition_testing_pointer = testing_analysis.get_pattern_recognition_testing_pointer();

	// Save results
	data_set.save("./data/neural_network/opennn/simple_pattern_recognition/data_set.xml");
	data_set.load("./data/neural_network/opennn/simple_pattern_recognition/data_set.xml");

	neural_network.save("./data/neural_network/opennn/simple_pattern_recognition/neural_network.xml");
	neural_network.load("./data/neural_network/opennn/simple_pattern_recognition/neural_network.xml");

	neural_network.save_expression("./data/neural_network/opennn/simple_pattern_recognition/expression.txt");

	performance_functional.save("./data/neural_network/opennn/simple_pattern_recognition/performance_functional.xml");
	performance_functional.load("./data/neural_network/opennn/simple_pattern_recognition/performance_functional.xml");

	training_strategy.save("./data/neural_network/opennn/simple_pattern_recognition/training_strategy.xml");
	training_strategy.load("./data/neural_network/opennn/simple_pattern_recognition/training_strategy.xml");

	training_stategy_results.save("./data/neural_network/opennn/simple_pattern_recognition/training_strategy_results.dat");

	pattern_recognition_testing_pointer->save_binary_classification_test("./data/neural_network/opennn/simple_pattern_recognition/binary_classification_test.dat");
	pattern_recognition_testing_pointer->save_confusion("./data/neural_network/opennn/simple_pattern_recognition/confusion.dat");
}

}  // namespace my_opennn
