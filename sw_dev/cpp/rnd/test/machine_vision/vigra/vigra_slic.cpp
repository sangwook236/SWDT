#include <vigra/slic.hxx>
#include <vigra/colorconversions.hxx>
#include <vigra/multi_array.hxx>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

// [ref] create_superpixel_boundary() in ${CPP_RND_HOME}/test/segmentation/gslic/gslic_main.cpp
void create_superpixel_boundary(const cv::Mat &superpixel_mask, cv::Mat &superpixel_boundary)
{
	superpixel_boundary = cv::Mat::zeros(superpixel_mask.size(), CV_8UC1);

	for (int i = 1; i < superpixel_mask.rows - 1; ++i)
		for (int j = 1; j < superpixel_mask.cols - 1; ++j)
		{
			const int idx = superpixel_mask.at<int>(i, j);
			if (idx != superpixel_mask.at<int>(i, j - 1) || idx != superpixel_mask.at<int>(i, j + 1) ||
				idx != superpixel_mask.at<int>(i - 1, j) || idx != superpixel_mask.at<int>(i + 1, j))
				superpixel_boundary.at<unsigned char>(i, j) = 255;
		}
}

// [ref] ${VIGRA_HOME}/doc/vigra/group__SeededRegionGrowing.html
void slic_example()
{
#if 0
	const std::string input_filename("./data/segmentation/beach.png");
	const std::string output_filename("./data/segmentation/beach_segmented.png");
#elif 0
	const std::string input_filename("./data/segmentation/grain.png");
	const std::string output_filename("./data/segmentation/grain_segmented.png");
#elif 1
	const std::string input_filename("./data/segmentation/brain_small.png");
	const std::string output_filename("./data/segmentation/brain_small_segmented.png");
#endif

	cv::Mat input_img = cv::imread(input_filename, cv::IMREAD_COLOR);
	if (input_img.empty())
	{
		std::cout << "image file not found: " << input_filename << std::endl;
		return;
	}

	vigra::MultiArray<2, vigra::RGBValue<float> > src(vigra::Shape2(input_img.rows, input_img.cols));
	for (int i = 0; i < input_img.rows; ++i)
		for (int j = 0; j < input_img.cols; ++j)
		{
			const cv::Vec3b &bgr = input_img.at<cv::Vec3b>(i, j);
			src(i, j).red() = (float)bgr[2];
			src(i, j).green() = (float)bgr[1];
			src(i, j).blue() = (float)bgr[0];
		}

	// transform image to Lab color space.
	vigra::transformMultiArray(vigra::srcMultiArrayRange(src), vigra::destMultiArray(src), vigra::RGBPrime2LabFunctor<float>());

	vigra::MultiArray<2, unsigned int> labels(src.shape());
	{
		const int seedDistance = 15;
		const double intensityScaling = 20.0;

		boost::timer::auto_cpu_timer timer;
		// compute seeds automatically, perform 40 iterations,
		// and scale intensity differences down to 1/20 before comparing with spatial distances.
		vigra::slicSuperpixels(src, labels, intensityScaling, seedDistance, vigra::SlicOptions().iterations(40));
	}

	// display.
	cv::Mat label_mask(input_img.size(), CV_32SC1, cv::Scalar::all(0));
	for (int i = 0; i < label_mask.rows; ++i)
		for (int j = 0; j < label_mask.cols; ++j)
			label_mask.at<int>(i, j) = labels(i, j);

	cv::Mat label_img;
	create_superpixel_boundary(label_mask, label_img);

	cv::imshow("SLIC - input", input_img);
	cv::imshow("SLIC - label", label_img);

	cv::waitKey(0);

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_vigra {

void slic()
{
	local::slic_example();
}

}  // namespace my_vigra
