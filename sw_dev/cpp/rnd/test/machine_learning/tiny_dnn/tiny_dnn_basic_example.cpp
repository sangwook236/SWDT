#include <tiny_dnn/tiny_dnn.h>
#include <iostream>
#include <algorithm>
#include <memory>


namespace my_tiny_dnn {

// REF [file] >> sample1_convnet() in ${TINY_DNN_HOME}/examples/main.cpp
// Learning convolutional neural networks (LeNet-5 like architecture).
void convnet_sample(const std::string& data_dir_path)
{
    // Construct LeNet-5 architecture.
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;

    // Connection table [Y.Lecun, 1998 Table.1].
#define O true
#define X false
    static const bool connection[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

	// REF [paper] >> "Gradient-Based Learning Applied to Document Recognition", PIEEE 1998.
	// Construct nets.
	nn << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(32, 32, 5, 1, 6)  // 32x32 in, 5x5 kernel, 1-6 fmaps conv.
       << tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(28, 28, 6, 2)  // 28x28 in, 6 fmaps, 2x2 subsampling.
       << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(14, 14, 5, 6, 16, tiny_dnn::core::connection_table(connection, 6, 16))  // With connection-table.
       << tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(10, 10, 16, 2)
       << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(5, 5, 5, 16, 120)
       << tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h>(120, 10);

    std::cout << "Load models..." << std::endl;

    // Load MNIST dataset.
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

	std::cout << "Start training..." << std::endl;
	
	tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;
    int minibatch_size = 10;

	optimizer.alpha *= std::sqrt(minibatch_size);

    // Create callback.
    auto on_enumerate_epoch = [&]()
	{
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_dnn::result res = nn.test(test_images, test_labels);

        std::cout << optimizer.alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]()
	{ 
        disp += minibatch_size; 
    };
    
    // Training.
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "End training..." << std::endl;

	std::cout << "Start testing..." << std::endl;

	// Test and show results.
	nn.test(test_images, test_labels).print_detail(std::cout);

	std::cout << "Eend testing..." << std::endl;

    // Save networks.
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

// REF [file] >> sample2_mlp() in ${TINY_DNN_HOME}/examples/main.cpp
// Learning 3-Layer Networks.
void mlp_sample(const std::string& data_dir_path)
{
    const tiny_dnn::cnn_size_t num_hidden_units = 500;

#if defined(_MSC_VER) && _MSC_VER < 1800
    // Initializer-list is not supported.
    int num_units[] = { 28 * 28, num_hidden_units, 10 };
    auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>(num_units, num_units + 3);
#else
    auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>({ 28 * 28, num_hidden_units, 10 });
#endif
	tiny_dnn::gradient_descent optimizer;

    // Load MNIST dataset.
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

	optimizer.alpha = 0.001;

	std::cout << "Start training..." << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    // Create callback.
    auto on_enumerate_epoch = [&]()
	{
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_dnn::result res = nn.test(test_images, test_labels);

        std::cout << optimizer.alpha << "," << res.num_success << "/" << res.num_total << std::endl;

		optimizer.alpha *= 0.85;  // Decay learning rate.
		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&]()
	{ 
        ++disp; 
    };  

	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);

	std::cout << "End training..." << std::endl;
}

// REF [file] >> sample3_dae() in ${TINY_DNN_HOME}/examples/main.cpp
// Denoising auto-encoder.
void denoising_auto_encoder_sample(const std::string& data_dir_path)
{
#if defined(_MSC_VER) && _MSC_VER < 1800
    // Initializer-list is not supported.
	//int num_units[] = { 100, 400, 100 };
	int num_units[] = { 28*28, 400, 28*28 };  // For MNIST dataset.
	//int num_units[] = { 28*28, 400, 100, 400, 28*28 };  // For MNIST dataset.
	auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>(num_units, num_units + 3);
#else
	//auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>({ 100, 400, 100 });
	auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>({ 28*28, 400, 28*28 });  // For MNIST dataset.
	//auto nn = tiny_dnn::make_mlp<tiny_dnn::activation::tan_h>({ 28*28, 400, 100, 400, 28*28 });  // For MNIST dataset.
#endif

    // Load train-data.
	//std::vector<tiny_dnn::vec_t> train_data_original;
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

	//std::vector<tiny_dnn::vec_t> train_data_corrupted(train_data_original);
	std::vector<tiny_dnn::vec_t> train_data_corrupted(train_images);  // For MNIST dataset.

    for (auto& d : train_data_corrupted)
        d = tiny_dnn::corrupt(std::move(d), 0.1, 0.0);  // Corrupt 10% data.

	tiny_dnn::gradient_descent optimizer;

	//
	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;

	// Create callback.
	auto on_enumerate_epoch = [&]()
	{
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_dnn::result res = nn.test(test_images, test_labels);

		std::cout << optimizer.alpha << "," << res.num_success << "/" << res.num_total << std::endl;

		optimizer.alpha *= 0.85;  // Decay learning rate.
		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_data = [&]()
	{
		++disp;
	};

	std::cout << "start training..." << std::endl;

	// Learning 100-400-100 denoising auto-encoder.
	//nn.train<tiny_dnn::mse>(optimizer, train_data_corrupted, train_data_original);
	// Learning 28*28-400-28*28 denoising auto-encoder.
	nn.train<tiny_dnn::mse>(optimizer, train_data_corrupted, train_labels, 1, 100, on_enumerate_data, on_enumerate_epoch);  // for MNIST dataset.

	std::cout << "end training..." << std::endl;
}

// REF [file] >> sample4_dropout() in ${TINY_DNN_HOME}/examples/main.cpp
// Dropout-learning.
void dropout_sample(const std::string& data_dir_path)
{
    typedef tiny_dnn::network<tiny_dnn::sequential> Network;
    Network nn;
    tiny_dnn::cnn_size_t input_dim = 28*28;
    tiny_dnn::cnn_size_t hidden_units = 800;
    tiny_dnn::cnn_size_t output_dim = 10;
	tiny_dnn::gradient_descent optimizer;

	tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h> f1(input_dim, hidden_units);
	tiny_dnn::dropout_layer dropout(hidden_units, 0.5);
	tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h> f2(hidden_units, output_dim);
    nn << f1 << dropout << f2;

	optimizer.alpha = 0.003;  // TODO: not optimized.
	optimizer.lambda = 0.0;

    // Load MNIST dataset.
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

	std::cout << "Start training..." << std::endl;

    // Load train-data & label_data.
    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;

    // Create callback.
    auto on_enumerate_epoch = [&]()
	{
        std::cout << t.elapsed() << "s elapsed." << std::endl;
  
        dropout.set_context(tiny_dnn::net_phase::test);
        tiny_dnn::result res = nn.test(test_images, test_labels);
        dropout.set_context(tiny_dnn::net_phase::train);

        std::cout << optimizer.alpha << "," << res.num_success << "/" << res.num_total << std::endl;

		optimizer.alpha *= 0.99;  // Decay learning rate.
		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&]()
	{
        ++disp;
    };

	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, 1, 100, on_enumerate_data, on_enumerate_epoch);

	std::cout << "End training..." << std::endl;

    // Change context to enable all hidden-units.
    //f1.set_context(tiny_dnn::dropout::test_phase);
    //std::cout << res.num_success << "/" << res.num_total << std::endl;
}

}  // namespace my_tiny_dnn
