//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// REF [file] >> ${SWDT_CPP_HOME}/rnd/test/machine_vision/opencv/opencv_util.cpp
void snake(cv::Mat &srcImage, cv::Mat &grayImage);

void active_contour_model()
{
	std::list<std::string> filenames;
	filenames.push_back("../data/machine_vision/opencv/pic1.png");
	filenames.push_back("../data/machine_vision/opencv/pic2.png");
	filenames.push_back("../data/machine_vision/opencv/pic3.png");
	filenames.push_back("../data/machine_vision/opencv/pic4.png");
	filenames.push_back("../data/machine_vision/opencv/pic5.png");
	filenames.push_back("../data/machine_vision/opencv/pic6.png");
	filenames.push_back("../data/machine_vision/opencv/stuff.jpg");
	filenames.push_back("../data/machine_vision/opencv/synthetic_face.png");
	filenames.push_back("../data/machine_vision/opencv/puzzle.png");
	filenames.push_back("../data/machine_vision/opencv/fruits.jpg");
	filenames.push_back("../data/machine_vision/opencv/lena_rgb.bmp");
	filenames.push_back("../data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_24.jpg");

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {

		cv::Mat srcImage = cv::imread(*it);
		if (srcImage.empty())
		{
			std::cout << "Failed to load an image: " << *it << std::endl;
			continue;
		}

		cv::Mat grayImage;
		if (1 == srcImage.channels())
			srcImage.copyTo(grayImage);
		else
			cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);

		//
		snake(srcImage, grayImage);

		//
		cv::imshow("Active Contour Model", srcImage);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
