//#include "stdafx.h"
#include <opennn/opennn.h>
#include <iostream>
#include <sstream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opennn {

// REF [file] >> ${OPENNN_HOME}/examples/simple_pattern_recognition/main.cpp
void simple_pattern_recognition_example()
{
	std::cout << "OpenNN. Simple Pattern Recognition Application." << std::endl;

	//std::srand((unsigned)std::time(NULL));

	// Data set object.
	OpenNN::DataSet data_set;

	data_set.set_data_file_name("./data/neural_network/opennn/simple_pattern_recognition.dat");

	data_set.load_data();

	OpenNN::Variables* variables_pointer = data_set.get_variables_pointer();

	variables_pointer->set_name(0, "x1");
	variables_pointer->set_name(1, "x2");
	variables_pointer->set_name(2, "y");

	// Neural network object.
	OpenNN::Instances* instances_pointer = data_set.get_instances_pointer();

	instances_pointer->set_training();

	OpenNN::Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
	OpenNN::Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

	const OpenNN::Vector<OpenNN::Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();


	OpenNN::NeuralNetwork neural_network(2, 2, 1);

	OpenNN::Inputs* inputs_pointer = neural_network.get_inputs_pointer();

	inputs_pointer->set_information(inputs_information);

	neural_network.construct_scaling_layer();

	OpenNN::ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

	scaling_layer_pointer->set_statistics(inputs_statistics);

	scaling_layer_pointer->set_scaling_method(OpenNN::ScalingLayer::NoScaling);

	OpenNN::MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();

	multilayer_perceptron_pointer->set_layer_activation_function(1, OpenNN::Perceptron::Logistic);

	OpenNN::Outputs* outputs_pointer = neural_network.get_outputs_pointer();

	outputs_pointer->set_information(targets_information);

	// Performance functional.
	OpenNN::PerformanceFunctional performance_functional(&neural_network, &data_set);

	// Training strategy
	OpenNN::TrainingStrategy training_strategy(&performance_functional);

	OpenNN::QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

	quasi_Newton_method_pointer->set_minimum_performance_increase(1.0e-4);

	OpenNN::TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

	// Testing analysis.
	instances_pointer->set_testing();

	OpenNN::TestingAnalysis testing_analysis(&neural_network, &data_set);

	OpenNN::Vector<double> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

	OpenNN::Matrix<size_t> confusion = testing_analysis.calculate_confusion();

//	Matrix<double> ROC_curve = testing_analysis.calculate_ROC_curve();

	// Save results.
	scaling_layer_pointer->set_scaling_method(OpenNN::ScalingLayer::MinimumMaximum);

	data_set.save("./data/neural_network/opennn/simple_pattern_recognition/data_set.xml");

	neural_network.save("./data/neural_network/opennn/simple_pattern_recognition/neural_network.xml");
	neural_network.save_expression("./data/neural_network/opennn/simple_pattern_recognition/expression.txt");

	performance_functional.save("./data/neural_network/opennn/simple_pattern_recognition/performance_functional.xml");

	training_strategy.save("./data/neural_network/opennn/simple_pattern_recognition/training_strategy.xml");
	training_strategy_results.save("./data/neural_network/opennn/simple_pattern_recognition/training_strategy_results.dat");

	binary_classification_tests.save("./data/neural_network/opennn/simple_pattern_recognition/binary_classification_tests.dat");
	confusion.save("./data/neural_network/opennn/simple_pattern_recognition/confusion.dat");
}

}  // namespace my_opennn
