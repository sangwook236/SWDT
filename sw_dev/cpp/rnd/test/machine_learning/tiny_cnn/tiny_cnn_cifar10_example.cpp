#include <tiny_cnn/tiny_cnn.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>


namespace local {
namespace {

template <typename N>
void construct_net(N& nn)
{
    typedef tiny_cnn::convolutional_layer<tiny_cnn::activation::identity> conv;
    typedef tiny_cnn::max_pooling_layer<tiny_cnn::activation::relu> pool;

    const int n_fmaps = 32;  // number of feature maps for upper layer
    const int n_fmaps2 = 64;  // number of feature maps for lower layer
    const int n_fc = 64;  // number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, tiny_cnn::padding::same)
        << pool(32, 32, n_fmaps, 2)
        << conv(16, 16, 5, n_fmaps, n_fmaps, tiny_cnn::padding::same)
        << pool(16, 16, n_fmaps, 2)
        << conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_cnn::padding::same)
        << pool(8, 8, n_fmaps2, 2)
        << tiny_cnn::fully_connected_layer<tiny_cnn::activation::identity>(4 * 4 * n_fmaps2, n_fc)
        << tiny_cnn::fully_connected_layer<tiny_cnn::activation::softmax>(n_fc, 10);
}

void train_cifar10(const std::string& data_dir_path, const double learning_rate, std::ostream& log)
{
    // specify loss-function and learning strategy
	tiny_cnn::network<tiny_cnn::cross_entropy, tiny_cnn::adam> nn;

    construct_net(nn);

    log << "learning rate:" << learning_rate << std::endl;

    std::cout << "load models..." << std::endl;

    // load cifar dataset
    std::vector<tiny_cnn::label_t> train_labels, test_labels;
    std::vector<tiny_cnn::vec_t> train_images, test_images;

    for (int i = 1; i <= 5; ++i)
	{
		tiny_cnn::parse_cifar10(data_dir_path + "/data_batch_" + std::to_string(i) + ".bin", &train_images, &train_labels, -1.0, 1.0, 0, 0);
    }

	tiny_cnn::parse_cifar10(data_dir_path + "/test_batch.bin", &test_images, &test_labels, -1.0, 1.0, 0, 0);

    std::cout << "start learning" << std::endl;

	tiny_cnn::progress_display disp(train_images.size());
	tiny_cnn::timer t;
    const int n_minibatch = 10;  // minibatch size
    const int n_train_epochs = 30;  // training duration

    nn.optimizer().alpha *= std::sqrt(n_minibatch) * learning_rate;

    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        log << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() {
        disp += n_minibatch;
    };

    // training
    nn.train(train_images, train_labels, n_minibatch, n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("cifar-weights");
    ofs << nn;
}

}  // namespace local
}  // unnamed namespace

namespace my_tiny_cnn {

// REF [file] >> ${TINY_CNN_HOME}/examples/cifar10/train.cpp
void cifar10_train_example()
{
	const std::string path_to_data("./data/machine_learning");
	const double learning_rate = 0.01;

	local::train_cifar10(path_to_data, learning_rate, std::cout);
}

}  // namespace my_tiny_cnn
