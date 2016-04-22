#include <tiny_cnn/tiny_cnn.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


namespace local {
namespace {

// rescale output to 0-100.
template <typename Activation>
double rescale(double x)
{
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// convert tiny_cnn::image to cv::Mat and resize.
cv::Mat image2mat(tiny_cnn::image<>& img)
{
	cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
	cv::Mat resized;
	cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
	return resized;
}

void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, tiny_cnn::vec_t& data)
{
	auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
	if (nullptr == img.data) return;  // cannot open, or it's not an image.

	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));

	// mnist dataset is "white on black", so negate required.
	std::transform(resized.begin(), resized.end(), std::back_inserter(data), [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void construct_net(tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad>& nn)
{
    // connection table [Y.Lecun, 1998 Table.1].
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

	// REF [paper] >> "Gradient-Based Learning Applied to Document Recognition", PIEEE 1998.
    // construct nets.
    nn << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out.
       << tiny_cnn::average_pooling_layer<tiny_cnn::activation::tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out.
       << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(14, 14, 5, 6, 16, tiny_cnn::connection_table(tbl, 6, 16))  // C3, 6@14x14-in, 16@10x10-in.
       << tiny_cnn::average_pooling_layer<tiny_cnn::activation::tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out.
       << tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out.
       << tiny_cnn::fully_connected_layer<tiny_cnn::activation::tan_h>(120, 10);       // F6, 120-in, 10-out.
}

void train_lenet(const std::string& data_dir_path)
{
    // specify loss-function and learning strategy.
	tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad> nn;

    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset.
    std::vector<tiny_cnn::label_t> train_labels, test_labels;
    std::vector<tiny_cnn::vec_t> train_images, test_images;

	tiny_cnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_cnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_cnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

	tiny_cnn::progress_display disp(train_images.size());
	tiny_cnn::timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback.
    auto on_enumerate_epoch = [&]()
	{
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]()
	{
        disp += minibatch_size;
    };

    // training.
    nn.train(train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results.
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks.
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

void recognize(const std::string& dictionary, const std::string& filename)
{
	tiny_cnn::network<tiny_cnn::mse, tiny_cnn::adagrad> nn;

	construct_net(nn);

	// load nets.
	std::ifstream ifs(dictionary.c_str());
	ifs >> nn;

	// convert imagefile to vec_t.
	tiny_cnn::vec_t data;
	convert_image(filename, -1.0, 1.0, 32, 32, data);

	// recognize.
	auto res = nn.predict(data);
	std::vector<std::pair<double, int> > scores;

	// sort & print top-3.
	for (int i = 0; i < 10; ++i)
		scores.emplace_back(rescale<tiny_cnn::activation::tan_h>(res[i]), i);

	std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

	for (int i = 0; i < 3; i++)
		std::cout << scores[i].second << "," << scores[i].first << std::endl;

	// visualize outputs of each layer.
	for (std::size_t i = 0; i < nn.depth(); ++i)
	{
		auto out_img = nn[i]->output_to_image();
		cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
	}
	// visualize filter shape of first convolutional layer.
	auto weight = nn.at<tiny_cnn::convolutional_layer<tiny_cnn::activation::tan_h> >(0).weight_to_image();
	cv::imshow("weights:", image2mat(weight));

	cv::waitKey(0);
}

}  // namespace local
}  // unnamed namespace

namespace my_tiny_cnn {

// REF [file] >> ${TINY_CNN_HOME}/examples/mnist/train.cpp
void mnist_train_example()
{
	// REF [site] >> http://yann.lecun.com/exdb/mnist/
	const std::string path_to_data("./data/machine_learning/mnist");

	local::train_lenet(path_to_data);
}

// REF [file] >> ${TINY_CNN_HOME}/examples/mnist/test.cpp
void mnist_test_example()
{
	try
	{
		const std::string image_file("./data/machine_learning/mnist/???.???");

		local::recognize("LeNet-weights", image_file);
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

}  // namespace my_tiny_cnn
