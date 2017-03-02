#include <tiny_dnn/tiny_dnn.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


namespace local {
namespace {

// Rescale output to 0-100.
template <typename Activation>
double rescale(double x)
{
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// Convert tiny_dnn::image to cv::Mat and resize.
cv::Mat image2mat(tiny_dnn::image<>& img)
{
	cv::Mat ori(static_cast<int>(img.height()), static_cast<int>(img.width()), CV_8U, &img.at(0, 0));
	cv::Mat resized;
	cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
	return resized;
}

void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, tiny_dnn::vec_t& data)
{
	auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
	if (nullptr == img.data) return;  // Cannot open, or it's not an image.

	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));

	// MNIST dataset is "white on black", so negate required.
	std::transform(resized.begin(), resized.end(), std::back_inserter(data), [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void construct_net(tiny_dnn::network<tiny_dnn::sequential>& nn)
{
    // Connection table [Y.Lecun, 1998 Table.1].
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::backend_t::tiny_dnn;

	// REF [paper] >> "Gradient-Based Learning Applied to Document Recognition", PIEEE 1998.
    // Construct nets.
    nn << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(32, 32, 5, 1, 6, tiny_dnn::core::padding::valid, true, 1, 1, backend_type)  // C1, 1@32x32-in, 6@28x28-out.
       << tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(28, 28, 6, 2)  // S2, 6@28x28-in, 6@14x14-out.
       << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(14, 14, 5, 6, 16, tiny_dnn::core::connection_table(tbl, 6, 16), tiny_dnn::core::padding::valid, true, 1, 1, backend_type)  // C3, 6@14x14-in, 16@10x10-in.
       << tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out.
       << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(5, 5, 5, 16, 120, tiny_dnn::core::padding::valid, true, 1, 1, backend_type)  // C5, 16@5x5-in, 120@1x1-out.
       << tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h>(120, 10, true, backend_type);  // F6, 120-in, 10-out.
}

void train_lenet(const std::string& dataset_dir_path)
{
    // Specify loss-function and learning strategy.
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;

    construct_net(nn);

    std::cout << "Load models..." << std::endl;

    // Load MNIST dataset.
    std::vector<tiny_dnn::label_t> train_labels, test_labels;
    std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(dataset_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(dataset_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(dataset_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(dataset_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "Start training." << std::endl;

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

	optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

    // Create callback.
    auto on_enumerate_epoch = [&]()
	{
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart((unsigned long)train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]()
	{
        disp += minibatch_size;
    };

    // Training.
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "End training." << std::endl;

	//
	std::cout << "Start testing." << std::endl;

	// Test and show results.
	nn.test(test_images, test_labels).print_detail(std::cout);

	std::cout << "End testing." << std::endl;

	// Save network model & trained weights.
#if 0
	std::ofstream ofs("./data/machine_learning/tiny_dnn/LeNet-model");
	ofs << nn;
#else
	nn.save("./data/machine_learning/tiny_dnn/LeNet-model");
#endif
}

void recognize(const std::string& dictionary, const std::string& filename)
{
	tiny_dnn::network<tiny_dnn::sequential> nn;

	nn.load(dictionary);

	// Load nets.
#if 1
	std::ifstream ifs(dictionary.c_str());
	ifs >> nn;
#else
	nn.load(dictionary);
#endif

	// Convert imagefile to vec_t.
	tiny_dnn::vec_t data;
	convert_image(filename, -1.0, 1.0, 32, 32, data);

	// Recognize.
	auto res = nn.predict(data);
	std::vector<std::pair<double, int> > scores;

	// Sort & print top-3.
	for (int i = 0; i < 10; ++i)
		scores.emplace_back(rescale<tiny_dnn::activation::tan_h>(res[i]), i);

	std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int> >());

	for (int i = 0; i < 3; ++i)
		std::cout << scores[i].second << "," << scores[i].first << std::endl;

	// Visualize outputs of each layer.
	for (size_t i = 0; i < nn.layer_size(); ++i)
	{
		auto out_img = nn[i]->output_to_image();
		cv::imshow("Layer:" + std::to_string(i), image2mat(out_img));
	}
	// Visualize filter shape of first convolutional layer.
	auto weight = nn.at<tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h> >(0).weight_to_image();
	cv::imshow("Weights:", image2mat(weight));

	cv::waitKey(0);
}

}  // namespace local
}  // unnamed namespace

namespace my_tiny_dnn {

// REF [file] >> ${TINY_DNN_HOME}/examples/mnist/train.cpp
void mnist_train_example()
{
	// REF [site] >> http://yann.lecun.com/exdb/mnist/
	const std::string path_to_dataset("D:/dataset/pattern_recognition/mnist");

	local::train_lenet(path_to_dataset);
}

// REF [file] >> ${TINY_DNN_HOME}/examples/mnist/test.cpp
void mnist_test_example()
{
	try
	{
		const std::string image_file("./data/machine_learning/mnist/five1.png");
		//const std::string image_file("./data/machine_learning/mnist/zero1.png");
		//const std::string image_file("./data/machine_learning/mnist/four1.png");
		//const std::string image_file("./data/machine_learning/mnist/one1.png");

		local::recognize("./data/machine_learning/tiny_dnn/LeNet-model", image_file);
	}
	catch (const cv::Exception& e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;
	}
}

}  // namespace my_tiny_dnn
