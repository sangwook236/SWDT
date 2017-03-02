#if defined(_WIN64) || defined(WIN64)
#define DLIB_USE_CUDA 1
#elif defined(_WIN32) || defined(WIN32)
#else
#define DLIB_USE_CUDA 1
#endif
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include <vector>
#include <string>


namespace {
namespace local {

// Inception layer has some different convolutions inside.
template <typename SUBNET> using block_a1 = dlib::relu<dlib::con<10, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using block_a2 = dlib::relu<dlib::con<10, 3, 3, 1, 1, dlib::relu<dlib::con<16, 1, 1, 1, 1, SUBNET>>>>;
template <typename SUBNET> using block_a3 = dlib::relu<dlib::con<10, 5, 5, 1, 1, dlib::relu<dlib::con<16, 1, 1, 1, 1, SUBNET>>>>;
template <typename SUBNET> using block_a4 = dlib::relu<dlib::con<10, 1, 1, 1, 1, dlib::max_pool<3, 3, 1, 1, SUBNET>>>;

// Define a inception layer
template <typename SUBNET> using incept_a = dlib::inception4<block_a1, block_a2, block_a3, block_a4, SUBNET>;

// Network can have inception layers of different structure.
template <typename SUBNET> using block_b1 = dlib::relu<dlib::con<4, 1, 1, 1, 1, SUBNET>>;
template <typename SUBNET> using block_b2 = dlib::relu<dlib::con<4, 3, 3, 1, 1, SUBNET>>;
template <typename SUBNET> using block_b3 = dlib::relu<dlib::con<4, 1, 1, 1, 1, dlib::max_pool<3, 3, 1, 1, SUBNET>>>;
template <typename SUBNET> using incept_b = dlib::inception3<block_b1, block_b2, block_b3, SUBNET>;

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

// REF [file] >> ${DLIB_HOME}/examples/dnn_inception_ex.cpp
void dnn_inception_example()
{
	/*
		An inception network is composed of inception blocks of the form:

				   input from SUBNET
				  /        |        \
				 /         |         \
			  block1    block2  ... blockN
				 \         |         /
				  \        |        /
			  concatenate tensors from blocks
						   |
						output

		REF [paper] >> "Going deeper with convolutions." CVPR 2015.
	*/

	//
	const std::string mnist_data_dir("D:/dataset/pattern_recognition/mnist");

	// Load the MNIST dataset.
	std::vector<dlib::matrix<unsigned char>> training_images;
	std::vector<unsigned long> training_labels;
	std::vector<dlib::matrix<unsigned char>> testing_images;
	std::vector<unsigned long> testing_labels;
	dlib::load_mnist_dataset(mnist_data_dir, training_images, training_labels, testing_images, testing_labels);

	// Define a simple network for classifying MNIST digits.
	using net_type = dlib::loss_multiclass_log<
		dlib::fc<10,
		dlib::relu<dlib::fc<32,
		dlib::max_pool<2, 2, 2, 2, local::incept_b<
		dlib::max_pool<2, 2, 2, 2, local::incept_a<
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>;

	// Make an instance of our inception network.
	net_type net;
	std::cout << "The net has " << net.num_layers << " layers in it." << std::endl;
	std::cout << net << std::endl;

	std::cout << "Traning NN..." << std::endl;
	dlib::dnn_trainer<net_type> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.00001);
	trainer.set_mini_batch_size(128);
	trainer.be_verbose();
	trainer.set_synchronization_file("./data/general_purpose_library/dlib/inception_sync", std::chrono::seconds(20));
	// Train the network.
	trainer.train(training_images, training_labels);

	// Note that, since the trainer has been running images through the network, net will have a bunch of state in it related to the last batch of images it processed (e.g. outputs from each layer).
	// Since we don't care about saving that kind of stuff to disk we can tell the network to forget about that kind of transient data so that our file will be smaller.
	net.clean();
	dlib::serialize("./data/general_purpose_library/dlib/mnist_network_inception.dat") << net;
	// Recall the network from disk.
	//dlib::deserialize("./data/general_purpose_library/dlib/mnist_network_inception.dat") >> net;

	// Run the training images through the network.
	std::vector<unsigned long> predicted_labels = net(training_images);
	int num_right = 0;
	int num_wrong = 0;
	// See if it classified them correctly.
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

	// See if the network can correctly classify the testing images.
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

}  // namespace my_dlib
