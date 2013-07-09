//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cassert>


namespace {
namespace local {

void basic_operation_1()
{
	// mean & standard deviation
	{
		const std::string input_filename("./machine_vision_data/opencv/lena_rgb.bmp");

		const cv::Mat &img = cv::imread(input_filename, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cerr << "file not found: " << input_filename << std::endl;
			return;
		}

		const cv::Scalar mean1 = cv::mean(img);
		std::cout << "mean = " << mean1[0] << ", " << mean1[1] << ", " << mean1[2] << ", " << mean1[3] << std::endl;

		const cv::Scalar mean2, stddev;
		cv::meanStdDev(img, mean2, stddev);
		std::cout << "mean = " << mean2[0] << ", " << mean2[1] << ", " << mean2[2] << ", " << mean2[3] << std::endl;
		std::cout << "std dev = " << stddev[0] << ", " << stddev[1] << ", " << stddev[2] << ", " << stddev[3] << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void basic_operation()
{
	local::basic_operation_1();
}

}  // namespace my_opencv
