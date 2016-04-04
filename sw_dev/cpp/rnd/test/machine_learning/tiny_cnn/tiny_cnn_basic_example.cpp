#include <tiny_cnn/tiny_cnn.h>
#include <iostream>
#include <algorithm>
#include <memory>


namespace my_tiny_cnn {

// REF [file] >> sample1_convnet() in ${TINY_CNN_HOME}/examples/main.cpp
// learning convolutional neural networks (LeNet-5 like architecture)
void convnet_sample(const std::string& data_dir_path)
{
    // construct LeNet-5 architecture
    tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent_levenberg_marquardt> nn;

    // connection table [Y.Lecun, 1998 Table.1]
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
	// construct nets
	nn << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << tiny_cnn::average_pooling_layer<tiny_cnn::activation::tan_h>(28, 28, 6, 2)  // 28x28 in, 6 fmaps, 2x2 subsampling
       << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(14, 14, 5, 6, 16, tiny_cnn::connection_table(connection, 6, 16)) // with connection-table
       << tiny_cnn::average_pooling_layer<tiny_cnn::activation::tan_h>(10, 10, 16, 2)
       << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(5, 5, 5, 16, 120)
       << tiny_cnn::fully_connected_layer<tiny_cnn::activation::tan_h>(120, 10);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<tiny_cnn::label_t> train_labels, test_labels;
    std::vector<tiny_cnn::vec_t> train_images, test_images;

	tiny_cnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_cnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start learning" << std::endl;

    tiny_cnn::progress_display disp(train_images.size());
    tiny_cnn::timer t;
    int minibatch_size = 10;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){ 
        disp += minibatch_size; 
    };
    
    // training
    nn.train(train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

// REF [file] >> sample2_mlp() in ${TINY_CNN_HOME}/examples/main.cpp
// learning 3-Layer Networks
void mlp_sample(const std::string& data_dir_path)
{
    const tiny_cnn::cnn_size_t num_hidden_units = 500;

#if defined(_MSC_VER) && _MSC_VER < 1800
    // initializer-list is not supported
    int num_units[] = { 28 * 28, num_hidden_units, 10 };
    auto nn = tiny_cnn::make_mlp<tiny_cnn::mse, tiny_cnn::gradient_descent, tiny_cnn::activation::tan_h>(num_units, num_units + 3);
#else
    auto nn = tiny_cnn::make_mlp<tiny_cnn::mse, tiny_cnn::gradient_descent, tiny_cnn::activation::tan_h>({ 28 * 28, num_hidden_units, 10 });
#endif

    // load MNIST dataset
    std::vector<tiny_cnn::label_t> train_labels, test_labels;
    std::vector<tiny_cnn::vec_t> train_images, test_images;

	tiny_cnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
	tiny_cnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    nn.optimizer().alpha = 0.001;
    
    tiny_cnn::progress_display disp(train_images.size());
    tiny_cnn::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;

        tiny_cnn::result res = nn.test(test_images, test_labels);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.85; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){ 
        ++disp; 
    };  

    nn.train(train_images, train_labels, 1, 20, on_enumerate_data, on_enumerate_epoch);
}

// REF [file] >> sample3_dae() in ${TINY_CNN_HOME}/examples/main.cpp
// denoising auto-encoder
void denoising_auto_encoder_sample()
{
#if defined(_MSC_VER) && _MSC_VER < 1800
    // initializer-list is not supported
    int num_units[] = { 100, 400, 100 };
    auto nn = tiny_cnn::make_mlp<tiny_cnn::mse, tiny_cnn::gradient_descent, tiny_cnn::activation::tan_h>(num_units, num_units + 3);
#else
    auto nn = tiny_cnn::make_mlp<tiny_cnn::mse, tiny_cnn::gradient_descent, tiny_cnn::activation::tan_h>({ 100, 400, 100 });
#endif

    std::vector<tiny_cnn::vec_t> train_data_original;

    // load train-data
    std::vector<tiny_cnn::vec_t> train_data_corrupted(train_data_original);

    for (auto& d : train_data_corrupted)
	{
        d = tiny_cnn::corrupt(move(d), 0.1, 0.0);  // corrupt 10% data
    }

    // learning 100-400-100 denoising auto-encoder
    nn.train(train_data_corrupted, train_data_original);
}

// REF [file] >> sample4_dropout() in ${TINY_CNN_HOME}/examples/main.cpp
// dropout-learning
void dropout_sample(const std::string& data_dir_path)
{
    typedef tiny_cnn::network<tiny_cnn::mse, tiny_cnn::gradient_descent> Network;
    Network nn;
    tiny_cnn::cnn_size_t input_dim = 28*28;
    tiny_cnn::cnn_size_t hidden_units = 800;
    tiny_cnn::cnn_size_t output_dim = 10;

	tiny_cnn::fully_connected_layer<tiny_cnn::activation::tan_h> f1(input_dim, hidden_units);
	tiny_cnn::dropout_layer dropout(hidden_units, 0.5);
	tiny_cnn::fully_connected_layer<tiny_cnn::activation::tan_h> f2(hidden_units, output_dim);
    nn << f1 << dropout << f2;

    nn.optimizer().alpha = 0.003; // TODO: not optimized
    nn.optimizer().lambda = 0.0;

    // load MNIST dataset
    std::vector<tiny_cnn::label_t> train_labels, test_labels;
    std::vector<tiny_cnn::vec_t> train_images, test_images;

	tiny_cnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 0, 0);
	tiny_cnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 0, 0);

    // load train-data, label_data
    tiny_cnn::progress_display disp(train_images.size());
    tiny_cnn::timer t;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
  
        dropout.set_context(tiny_cnn::net_phase::test);
        tiny_cnn::result res = nn.test(test_images, test_labels);
        dropout.set_context(tiny_cnn::net_phase::train);

        std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

        nn.optimizer().alpha *= 0.99; // decay learning rate
        nn.optimizer().alpha = std::max((tiny_cnn::float_t)0.00001, nn.optimizer().alpha);

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_data = [&](){
        ++disp;
    };

    nn.train(train_images, train_labels, 1, 100, on_enumerate_data, on_enumerate_epoch);

    // change context to enable all hidden-units
    //f1.set_context(dropout::test_phase);
    //std::cout << res.num_success << "/" << res.num_total << std::endl;
}

}  // namespace my_tiny_cnn
