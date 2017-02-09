#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include <vector>
#include <string>


namespace {
namespace local {

// REF [file] >> ${DLIB_HOME}/examples/dnn_introduction_ex.cpp
void dnn_introduction_examples()
{
	const std::string mnist_data_dir("D:/work/swdt_github/sw_dev/cpp/rnd/bin/data/machine_learning/mnist");

	// Load MNIST data.
	std::vector<dlib::matrix<unsigned char>> training_images;
	std::vector<unsigned long> training_labels;
	std::vector<dlib::matrix<unsigned char>> testing_images;
	std::vector<unsigned long> testing_labels;
	dlib::load_mnist_dataset(mnist_data_dir, training_images, training_labels, testing_images, testing_labels);

	// Define the LeNet.
	using net_type = dlib::loss_multiclass_log<
		dlib::fc<10,
		dlib::relu<dlib::fc<84,
		dlib::relu<dlib::fc<120,
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
		dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>>>>>;

	// Create a network instance.
	net_type net;
	// Train it using the MNIST data.
	// Use mini-batch stochastic gradient descent with an initial learning rate of 0.01.
	dlib::dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(128);
	trainer.be_verbose();

	// Since DNN training can take a long time, we can ask the trainer to save its state to a file named "mnist_sync" every 20 seconds.
	trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
	// Begin training.
	// Keep doing this until the learning rate has dropped below the min learning rate defined above or the maximum number of epochs as been executed (defaulted to 10000). 
	trainer.train(training_images, training_labels);

	// Save to disk.
	net.clean();
	dlib::serialize("mnist_network.dat") << net;
	// Load .from disk.
	//dlib::deserialize("mnist_network.dat") >> net;

	// Evaluate with the traing dataset.
	std::vector<unsigned long> predicted_labels = net(training_images);
	int num_right = 0;
	int num_wrong = 0;
	// And then let's see if it classified them correctly.
	for (size_t i = 0; i < training_images.size(); ++i)
	{
		if (predicted_labels[i] == training_labels[i])
			++num_right;
		else
			++num_wrong;

	}
	std::cout << "Training num_right: " << num_right << std::endl;
	std::cout << "Training num_wrong: " << num_wrong << std::endl;
	std::cout << "Training accuracy:  " << num_right / (double)(num_right + num_wrong) << std::endl;

	// Evaluate with the test dataset.
	predicted_labels = net(testing_images);
	num_right = 0;
	num_wrong = 0;
	for (size_t i = 0; i < testing_images.size(); ++i)
	{
		if (predicted_labels[i] == testing_labels[i])
			++num_right;
		else
			++num_wrong;

	}
	std::cout << "Testing num_right: " << num_right << std::endl;
	std::cout << "Testing num_wrong: " << num_wrong << std::endl;
	std::cout << "Testing accuracy:  " << num_right / (double)(num_right + num_wrong) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

void dnn_example()
{
	local::dnn_introduction_examples();
}

}  // namespace my_dlib
