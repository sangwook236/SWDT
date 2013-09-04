//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include "../nyu_depth_toolbox_v2_lib/cbf_windows.h"
#else
#include "../nyu_depth_toolbox_v2_lib/cbf.h"
#endif
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_nyu_depth_toolbox_v2 {

// [ref]
//	${NYU_Depth_Dataset_V2_HOME}/toolbox_nyu_depth_v2/fill_depth_cross_bf.m
//	${NYU_Depth_Dataset_V2_HOME}/toolbox_nyu_depth_v2/demo_fill_depth_cross_bf_test.m.
void depth_filling_cross_bilateral_filter_example()
{
#if 0
	const std::string color_input_filename("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211659.png");
	//const std::string color_input_filename("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211705.png");
	//const std::string color_input_filename("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211713.png");
	//const std::string color_input_filename("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211839.png");
	//const std::string color_input_filename("../../hw_interface/bin/data/kinect/kinect2_rgba_20130725T211842.png");

	const std::string depth_input_filename("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211659.png");
	//const std::string depth_input_filename("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211705.png");
	//const std::string depth_input_filename("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211713.png");
	//const std::string depth_input_filename("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211839.png");
	//const std::string depth_input_filename("../../hw_interface/bin/data/kinect/kinect2_depth_transformed_20130725T211842.png");

	cv::Mat gray_img;
	gray_img = cv::imread(color_input_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (gray_img.empty())
	{
		std::cerr << "color input file not found: " << color_input_filename << std::endl;
		return;
	}

	cv::Mat depth_img;
	{
		cv::Mat tmp = cv::imread(depth_input_filename, CV_LOAD_IMAGE_UNCHANGED);  // CV_16UC1
		if (tmp.empty())
		{
			std::cerr << "depth input file not found: " << depth_input_filename << std::endl;
			return;
		}

		double minVal, maxVal;
		cv::minMaxLoc(tmp, &minVal, &maxVal);
		tmp.convertTo(depth_img, CV_8UC1, 255.0 / maxVal, 0.0);
	}
#elif 1
	const std::string color_input_filename("./data/signal_processing/nyu_depth_toolbox/nyu_depth_v2_labeled_1_rgb.png");
	const std::string depth_input_filename("./data/signal_processing/nyu_depth_toolbox/nyu_depth_v2_labeled_1_depth_abs.png");
	const std::string depth_ref_filename("./data/signal_processing/nyu_depth_toolbox/nyu_depth_v2_labeled_1_depth_filled.png");

	cv::Mat gray_img;
	gray_img = cv::imread(color_input_filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (gray_img.empty())
	{
		std::cerr << "color input file not found: " << color_input_filename << std::endl;
		return;
	}

	cv::Mat depth_img = cv::imread(depth_input_filename, CV_LOAD_IMAGE_UNCHANGED);  // CV_8UC1
	if (depth_img.empty())
	{
		std::cerr << "depth input file not found: " << depth_input_filename << std::endl;
		return;
	}
#endif

	const std::string output_filename("./data/signal_processing/nyu_depth_toolbox/depth_filling_cross_bf_output.png");

	const double sigma_s[] = { 12.0, 5.0, 8.0 };  // space sigma.
	const double sigma_r[] = { 0.2, 0.08, 0.02 };  // range sigma.

	// FIXME [delete] >>
	cv::Mat mask = 0 == depth_img;
	cv::imshow("depth-filling cross bilateral filter - mask", mask);

	std::vector<unsigned char> gray_img_vec(gray_img.rows * gray_img.cols, 0);
	std::vector<unsigned char> depth_img_vec(depth_img.rows * depth_img.cols, 0);
	//std::vector<bool> depth_noise_mask(depth_img.rows * depth_img.cols, false);  // run-time error.
	boost::scoped_array<bool> depth_noise_mask(new bool [depth_img.rows * depth_img.cols]);
	{
		for (std::size_t r = 0; r < (std::size_t)depth_img.rows; ++r)
			for (std::size_t c = 0; c < (std::size_t)depth_img.cols; ++c)
			{
				// row-major.
				gray_img_vec[c * depth_img.rows + r] = gray_img.at<unsigned char>(r, c);  // [0, 255]
				depth_img_vec[c * depth_img.rows + r] = depth_img.at<unsigned char>(r, c);  // [0, 255]
				depth_noise_mask[c * depth_img.rows + r] = 0 == depth_img.at<unsigned char>(r, c);
			}
	}

	//---------------------------------------------------------------

	std::cout << "start filtering ... " << std::endl;

	cv::Mat output_gray_img(depth_img.size(), CV_8UC1);
	std::vector<unsigned char> output_gray_img_vec(depth_img.rows * depth_img.cols, 0);
	{
		boost::timer::auto_cpu_timer timer;

		// cross bilateral filtering.
#if defined(WIN32) || defined(_WIN32)
		// TODO [check] >>
		//	-. in order to change IMG_HEIGHT, IMG_WIDTH, and/or NUM_SCALES, macros have to be changed in ${CPP_RND_HOME}/test/signal_processing/nyu_depth_toolbox_v2_lib/cbf_windows.h.
		//		#define NUM_SCALES 3
		//		#define IMG_HEIGHT 640
		//		#define IMG_WIDTH 480

		//cbf::cbf(depth_img.data, gray_img.data, (bool *)&depth_noise_mask[0], output_gray_img.data, (double *)sigma_s, (double *)sigma_r);
		cbf::cbf((unsigned char *)&depth_img_vec[0], (unsigned char *)&gray_img_vec[0], (bool *)&depth_noise_mask[0], (unsigned char *)&output_gray_img_vec[0], (double *)sigma_s, (double *)sigma_r);
#else
        const int height = depth_img.rows;
        const int width = depth_img.cols;
        const unsigned num_scales = 3;  // the number of scales at which to perform the filtering.

		//cbf::cbf(height, width, depth_img.data, gray_img.data, (bool *)&depth_noise_mask[0], output_gray_img.data, num_scales, (double *)sigma_s, (double *)sigma_r);
		cbf::cbf(height, width, (unsigned char *)&depth_img_vec[0], (unsigned char *)&gray_img_vec[0], (bool *)&depth_noise_mask[0], (unsigned char *)&output_gray_img_vec[0], num_scales, (double *)sigma_s, (double *)sigma_r);
#endif
	}

	std::cout << "end filtering ..." << std::endl;

	//---------------------------------------------------------------

	{
		for (std::size_t r = 0; r < (std::size_t)depth_img.rows; ++r)
			for (std::size_t c = 0; c < (std::size_t)depth_img.cols; ++c)
			{
				// row-major.
				output_gray_img.at<unsigned char>(r, c) = output_gray_img_vec[c * depth_img.rows + r];
			}

		cv::imwrite(output_filename, output_gray_img);
	}

	{
		cv::imshow("depth-filling cross bilateral filter - input", gray_img);
		cv::imshow("depth-filling cross bilateral filter - depth", depth_img);
		cv::imshow("depth-filling cross bilateral filter - output", output_gray_img);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

}  // namespace my_nyu_depth_toolbox_v2
