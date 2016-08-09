//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
//#include <fann/doublefann.h>
#include <fann/floatfann.h>
//#include <fann/fixedfann.h>
#include <fann/fann_cpp.h>
#else
//#include <doublefann.h>
#include <floatfann.h>
//#include <fixedfann.h>
#include <fann_cpp.h>
#endif

#include <iostream>
#include <iomanip>


namespace {
namespace local {

// Callback function that simply prints the information to std::cout
int print_callback(FANN::neural_net &net, FANN::training_data &train, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, unsigned int epochs, void *user_data)
{
    std::cout << "Epochs     " << std::setw(8) << epochs << ". " << "Current Error: " << std::left << net.get_MSE() << std::right << std::endl;
    return 0;
}

// [ref] ${FANN_HOME}/examples/xor_sample.cpp
// Test function that demonstrates usage of the fann C++ wrapper
void xor_sample()
{
	std::cout << std::endl << "XOR test started." << std::endl;

	const float learning_rate = 0.7f;
	const unsigned int num_layers = 3;
	const unsigned int num_input = 2;
	const unsigned int num_hidden = 3;
	const unsigned int num_output = 1;
	const float desired_error = 0.001f;
	const unsigned int max_iterations = 300000;
	const unsigned int iterations_between_reports = 1000;

	std::cout << std::endl << "Creating network." << std::endl;

	FANN::neural_net net;
	net.create_standard(num_layers, num_input, num_hidden, num_output);

	net.set_learning_rate(learning_rate);

	net.set_activation_steepness_hidden(1.0);
	net.set_activation_steepness_output(1.0);

	net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

	// Set additional properties such as the training algorithm
	//net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

	// Output network type and parameters
	std::cout << std::endl << "Network Type                         :  ";
	switch (net.get_network_type())
	{
	case FANN::LAYER:
		std::cout << "LAYER" << std::endl;
		break;
	case FANN::SHORTCUT:
		std::cout << "SHORTCUT" << std::endl;
		break;
	default:
		std::cout << "UNKNOWN" << std::endl;
		break;
	}
	net.print_parameters();

	std::cout << std::endl << "Training network." << std::endl;

	FANN::training_data data;
	if (data.read_train_from_file("./data/neural_network/fann/xor.data"))
	{
		// Initialize and train the network with the data
		net.init_weights(data);

		std::cout << "Max Epochs " << std::setw(8) << max_iterations << ". " << "Desired Error: " << std::left << desired_error << std::right << std::endl;
		net.set_callback(print_callback, NULL);
		net.train_on_data(data, max_iterations, iterations_between_reports, desired_error);

		//
		std::cout << std::endl << "Testing network." << std::endl;

		for (unsigned int i = 0; i < data.length_train_data(); ++i)
		{
			// Run the network on the test data
			fann_type *calc_out = net.run(data.get_input()[i]);

			std::cout << "XOR test (" << std::showpos << data.get_input()[i][0] << ", "
				<< data.get_input()[i][1] << ") -> " << *calc_out
				<< ", should be " << data.get_output()[i][0] << ", "
				<< "difference = " << std::noshowpos
				<< fann_abs(*calc_out - data.get_output()[i][0]) << std::endl;
		}

		//
		std::cout << std::endl << "Saving network." << std::endl;

		// Save the network in floating point and fixed point
		net.save("./data/neural_network/fann/xor_float.net");
		unsigned int decimal_point = net.save_to_fixed("./data/neural_network/fann/xor_fixed.net");
		data.save_train_to_fixed("./data/neural_network/fann/xor_fixed.data", decimal_point);

		std::cout << std::endl << "XOR test completed." << std::endl;
	}
}

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, unsigned int epochs)
{
	std::cout << "Epochs     " << epochs << ". MSE: " << fann_get_MSE(ann) << ". Desired-MSE: " << desired_error << std::endl;
	return 0;
}

// [ref] ${FANN_HOME}/examples/xor_train.c
void xor_train()
{
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float)0;
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;

	std::cout << "Creating network." << std::endl;
	struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	struct fann_train_data *data = fann_read_train_from_file("./data/neural_network/fann/xor.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);

	std::cout << "Training network." << std::endl;
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	std::cout << "Testing network. " << fann_test_data(ann, data) << std::endl;

	for (unsigned int i = 0; i < fann_length_train_data(data); i++)
	{
		fann_type *calc_out = fann_run(ann, data->input[i]);
		std::cout << "XOR test (" << data->input[i][0] << ',' << data->input[i][1] << ") -> " << calc_out[0] << ", should be " << data->output[i][0] << ", difference=" << fann_abs(calc_out[0] - data->output[i][0]) << std::endl;
	}

	std::cout << "Saving network." << std::endl;

	fann_save(ann, "./data/neural_network/fann/xor_float.net");

	const unsigned int decimal_point = fann_save_to_fixed(ann, "./data/neural_network/fann/xor_fixed.net");
	fann_save_train_to_fixed(data, "./data/neural_network/fann/xor_fixed.data", decimal_point);

	std::cout << "Cleaning up." << std::endl;
	fann_destroy_train(data);
	fann_destroy(ann);
}

// [ref] ${FANN_HOME}/examples/xor_test.c
void xor_test()
{
	std::cout << "Creating network." << std::endl;

#ifdef FIXEDFANN
	struct fann *ann = fann_create_from_file("./data/neural_network/fann/xor_fixed.net");
#else
	struct fann *ann = fann_create_from_file("./data/neural_network/fann/xor_float.net");
#endif

	if (!ann)
	{
		std::cout << "Error creating ann --- ABORTING." << std::endl;
		return;
	}

	fann_print_connections(ann);
	fann_print_parameters(ann);

	std::cout << "Testing network." << std::endl;

#ifdef FIXEDFANN
	struct fann_train_data *data = fann_read_train_from_file("./data/neural_network/fann/xor_fixed.data");
#else
	struct fann_train_data *data = fann_read_train_from_file("./data/neural_network/fann/xor.data");
#endif

	for (unsigned int i = 0; i < fann_length_train_data(data); ++i)
	{
		fann_reset_MSE(ann);
		fann_type *calc_out = fann_test(ann, data->input[i], data->output[i]);
#ifdef FIXEDFANN
		std::cout << "XOR test (" << data->input[i][0] << ", " << data->input[i][1] << ") -> " << calc_out[0] << ", should be " << data->output[i][0] << ", difference=" << ((float)fann_abs(calc_out[0] - data->output[i][0]) / fann_get_multiplier(ann)) << std::endl;

		if ((float)fann_abs(calc_out[0] - data->output[i][0]) / fann_get_multiplier(ann) > 0.2f)
		{
			std::cout << "Test failed" << std::endl;
			return;
		}
#else
		std::cout << "XOR test (" << data->input[i][0] << ", " << data->input[i][1] << ") -> " << calc_out[0] << ", should be " << data->output[i][0] << ", difference=" << (float)fann_abs(calc_out[0] - data->output[i][0]) << std::endl;
#endif
	}

	std::cout << "Cleaning up." << std::endl;
	fann_destroy_train(data);
	fann_destroy(ann);
}

}  // namespace local
}  // unnamed namespace

namespace my_fann {

void xor_example()
{
	local::xor_sample();

	//local::xor_train();  // C-style
	//local::xor_test();  // C-style
}

}  // namespace my_fann
