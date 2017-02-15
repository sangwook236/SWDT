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

// Parameterize the "block" layer on a BN layer (nominally some kind of batch normalization).
template <
	int N,
	template <typename> class BN,
	int stride,
	typename SUBNET
>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

// Define the skip layer mechanism used in the residual network paper.
template <
	template <int, template <typename> class, int, typename> class block,
	int N,
	template <typename> class BN,
	typename SUBNET
>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

// residual_down creates a network structure like this:
/*
		input from SUBNET
			/     \
		/       \
	block     downsample(using avg_pool)
		\       /
			\     /
		add tensors (using add_prev2 which adds the output of tag2 with avg_pool's output)
			|
			output
*/
template <
	template <int, template <typename> class, int, typename> class block,
	int N,
	template <typename> class BN,
	typename SUBNET
>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

// Define 4 different residual blocks.
template <typename SUBNET> using res = dlib::relu<residual<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares = dlib::relu<residual<block, 8, dlib::affine, SUBNET>>;
template <typename SUBNET> using res_down = dlib::relu<residual_down<block, 8, dlib::bn_con, SUBNET>>;
template <typename SUBNET> using ares_down = dlib::relu<residual_down<block, 8, dlib::affine, SUBNET>>;

// Define a residual network building block that uses parametric ReLU units instead of regular ReLU.
template <typename SUBNET>
using pres = dlib::prelu<dlib::add_prev1<dlib::bn_con<dlib::con<8, 3, 3, 1, 1, dlib::prelu<dlib::bn_con<dlib::con<8, 3, 3, 1, 1, dlib::tag1<SUBNET>>>>>>>>;

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

// REF [file] >> ${DLIB_HOME}/examples/dnn_introduction2_ex.cpp
void dnn_resnet_example()
{
	// Define the building block of a residual network (ResNet).
	//	REF [paper] >> Figure 2 in "Deep Residual Learning for Image Recognition", arXiv 2015.

	//
	const std::string mnist_data_dir("../../rnd/bin/data/machine_learning/mnist");

	// Load MNIST data.
	std::vector<dlib::matrix<unsigned char>> training_images;
	std::vector<unsigned long> training_labels;
	std::vector<dlib::matrix<unsigned char>> testing_images;
	std::vector<unsigned long> testing_labels;
	load_mnist_dataset(mnist_data_dir, training_images, training_labels, testing_images, testing_labels);

	// One of the features of cuDNN is the option to use slower methods that use less RAM or faster methods that use a lot of RAM.
	dlib::set_dnn_prefer_smallest_algorithms();

	// Define a residual network.
	const unsigned long number_of_classes = 10;
	using net_type = dlib::loss_multiclass_log<dlib::fc<number_of_classes,
		dlib::avg_pool_everything<
		local::res<local::res<local::res<local::res_down<
		dlib::repeat<9, local::res, // Repeat this layer 9 times.
		local::res_down<
		local::res<
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>>>;

	// Create a network.
	net_type net;
	// Use the same network but override the number of outputs at runtime you can do so like this:
	net_type net2(dlib::num_fc_outputs(15));

	// Replace some of the relu layers with prelu layers.
	using net_type2 = dlib::loss_multiclass_log<dlib::fc<number_of_classes,
		dlib::avg_pool_everything<
		local::pres<local::res<local::res<local::res_down<  // 2 prelu layers here.
		dlib::tag4<dlib::repeat<9, local::pres,  // 9 groups, each containing 2 prelu layers  .
		local::res_down<
		local::res<
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>>>>;

	// prelu layers have a floating point parameter.
	net_type2 pnet(
		dlib::prelu_(0.2),
		dlib::prelu_(0.25),
		// Initialize all the prelu instances in the repeat layer.
		// repeat_group() is needed to group the things that are part of repeat's block.
		dlib::repeat_group(dlib::prelu_(0.3), dlib::prelu_(0.4))
	);

	// Print the details of the pnet to the screen and inspect it.
	std::cout << "The pnet has " << pnet.num_layers << " layers in it." << std::endl;
	std::cout << pnet << std::endl;

	// Access layers individually using layer<index>(pnet).
	dlib::layer<3>(pnet).get_output();
	// Print the prelu parameter for layer 7.
	std::cout << "prelu param: " << dlib::layer<7>(pnet).layer_details().get_initial_param_value() << std::endl;

	// Access layers by their type.
	dlib::layer<dlib::tag1>(pnet);
	// Index relative to a tag.
	dlib::layer<dlib::tag4, 1>(pnet);  // Equivalent to layer<38+1>(pnet).
	// Access the layer 2 layers after tag4.
	dlib::layer<dlib::tag4, 2>(pnet);

	// The dnn_trainer will use SGD by default, but you can tell it to use different solvers like adam with a weight decay of 0.0005 and the given momentum parameters.
	dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0005, 0.9, 0.999));
	// If you have multiple graphics cards you can tell the trainer to use them together to make the training faster.
	// Use GPU cards 0 and 1.
	//dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0005, 0.9, 0.999), {0, 1});

	trainer.be_verbose();
	trainer.set_synchronization_file("./data/general_purpose_library/dlib/mnist_resnet_sync", std::chrono::seconds(100));

	// While the trainer is running it keeps an eye on the training error.
	// If it looks like the error hasn't decreased for the last 2000 iterations it will automatically reduce the learning rate by 0.1.
	trainer.set_iterations_without_progress_threshold(2000);
	trainer.set_learning_rate_shrink_factor(0.1);
	// The learning rate will start at 1e-3.
	trainer.set_learning_rate(1e-3);

	// What if your training dataset is so big it doesn't fit in RAM?
	// You make mini-batches yourself, any way you like, and you send them to the trainer by repeatedly calling trainer.train_one_step().

	std::vector<dlib::matrix<unsigned char>> mini_batch_samples;
	std::vector<unsigned long> mini_batch_labels;
	dlib::rand rnd(std::time(0));
	// Loop until the trainer's automatic shrinking has shrunk the learning rate to 1e-6.
	// Given our settings, this means it will stop training after it has shrunk the learning rate 3 times.
	while (trainer.get_learning_rate() >= 1e-6)
	{
		mini_batch_samples.clear();
		mini_batch_labels.clear();

		// Make a 128 image mini-batch.
		while (mini_batch_samples.size() < 128)
		{
			auto idx = rnd.get_random_32bit_number() % training_images.size();
			mini_batch_samples.push_back(training_images[idx]);
			mini_batch_labels.push_back(training_labels[idx]);
		}

		// Tell the trainer to update the network given this mini-batch
		trainer.train_one_step(mini_batch_samples, mini_batch_labels);

		// You can also feed validation data into the trainer by periodically calling trainer.test_one_step(samples, labels).
		// Unlike train_one_step(), test_one_step() doesn't modify the network, it only computes the testing error which it records internally.
		// This testing error will then be print in the verbose logging and will also determine when the trainer's automatic learning rate shrinking happens.
		// Therefore, test_one_step() can be used to perform automatic early stopping based on held out data.
	}

	// When you call train_one_step(), the trainer will do its processing in a separate thread.
	// This allows the main thread to work on loading data while the trainer is busy executing the mini-batches in parallel.
	// However, this also means we need to wait for any mini-batches that are still executing to stop before we mess with the net object.
	// Calling get_net() performs the necessary synchronization.
	trainer.get_net();

	net.clean();
	dlib::serialize("./data/general_purpose_library/dlib/mnist_res_network.dat") << net;

	// Now we have a trained network.
	// However, it has batch normalization layers in it.
	// As is customary, we should replace these with simple affine layers before we use the network.
	using test_net_type = dlib::loss_multiclass_log<dlib::fc<number_of_classes,
		dlib::avg_pool_everything<
		local::ares<local::ares<local::ares<local::ares_down<
		dlib::repeat<9, local::ares,
		local::ares_down<
		local::ares<
		dlib::input<dlib::matrix<unsigned char>>
		>>>>>>>>>>;
	// Simply assign our trained net to our testing net.
	test_net_type tnet = net;
	// Deserialize it directly into your testing network.
	dlib::deserialize("./data/general_purpose_library/dlib/mnist_res_network.dat") >> tnet;

	// Run the testing network over our data.
	std::vector<unsigned long> predicted_labels = tnet(training_images);
	int num_right = 0;
	int num_wrong = 0;
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

	predicted_labels = tnet(testing_images);
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
