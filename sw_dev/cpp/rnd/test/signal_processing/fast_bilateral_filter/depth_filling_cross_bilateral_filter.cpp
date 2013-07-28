//#include "stdafx.h"
#define CHRONO
#include "../fast_bilateral_filter_lib/geom.h"
#include "../fast_bilateral_filter_lib/fast_lbf.h"
//#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_fast_bilateral_filter {

// [ref]
//	${NYU_Depth_Dataset_V2_HOME}/toolbox_nyu_depth_v2/fill_depth_cross_bf.m
//	${NYU_Depth_Dataset_V2_HOME}/toolbox_nyu_depth_v2/demo_fill_depth_cross_bf_test.m.
void depth_filling_cross_bilateral_filter()
{
	typedef Array_2D<double> image_type;

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

	const std::string output_filename("./signal_processing_data/fast_bilateral_filter/depth_filling_cross_bf_output.png");
	
	const double sigma_s = 16;  // space sigma.
	const double sigma_r = 0.2;  // range sigma.

	//---------------------------------------------------------------

	const std::size_t width = 640, height = 480;

	image_type image(width, height);
	cv::Mat color_img;
	{
		std::cout << "Load the color input image '" << color_input_filename << "'... " << std::flush;

		color_img = cv::imread(color_input_filename, CV_LOAD_IMAGE_COLOR);
		if (color_img.empty())
		{
			std::cerr << "color input file not found: " << color_input_filename << std::endl;
			return;
		}

		for (unsigned y = 0; y < height; ++y)
		{
			for (unsigned x = 0; x < width; ++x)
			{
				const cv::Vec3b rgb = color_img.at<cv::Vec3b>(y, x);
				image(x, y) = (20.0 * rgb[0] + 40.0 * rgb[1] + 1.0 * rgb[2]) / (61.0 * 255.0); 
			}
		}

		std::cout << "Done" << std::endl;
	}

	//---------------------------------------------------------------

	image_type depth(width, height);
	cv::Mat depth_img;
	{
		std::cout << "Load the depth input image '" << depth_input_filename << "'... " << std::flush;

		cv::Mat tmp = cv::imread(depth_input_filename, CV_LOAD_IMAGE_UNCHANGED);  // CV_16UC1
		if (tmp.empty())
		{
			std::cerr << "depth input file not found: " << depth_input_filename << std::endl;
			return;
		}

		double minVal, maxVal;
		cv::minMaxLoc(tmp, &minVal, &maxVal);
		tmp.convertTo(depth_img, CV_32FC1, 1.0 / maxVal, 0.0);

		for (unsigned y = 0; y < height; ++y)
			for (unsigned x = 0; x < width; ++x)
				depth(x, y) = depth_img.at<float>(y, x);

		std::cout << "Done" << std::endl;
	}

	std::cout << "sigma_s    = " << sigma_s << std::endl;
	std::cout << "sigma_r    = " << sigma_r << std::endl;

	//---------------------------------------------------------------

	std::cout << "Filter the image... " << std::endl;

	image_type filtered_image(width, height);
	Image_filter::fast_LBF(depth, image, sigma_s, sigma_r, false, &filtered_image, &filtered_image);

	std::cout << "Filtering done" << std::endl;

	//---------------------------------------------------------------

	cv::Mat output_gray_img(height, width, CV_8UC1);
	{
		std::cout << "Write the output image '" << output_filename << "'... " << std::flush;

		for (unsigned y = 0; y < height; ++y)
			for (unsigned x = 0; x < width; ++x)
				output_gray_img.at<unsigned char>(y, x) = static_cast<unsigned char>(Math_tools::clamp(0.0, 255.0, filtered_image(x, y) * 255.0));

		//cv::imwrite(output_filename, output_gray_img);

		std::cout << "Done" << std::endl;
	}

	//---------------------------------------------------------------

	{
		cv::imshow("depth-filling cross bilateral filter - input", color_img);
		cv::imshow("depth-filling cross bilateral filter - depth", depth_img);
		cv::imshow("depth-filling cross bilateral filter - output", output_gray_img);

		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}

}  // namespace my_fast_bilateral_filter
