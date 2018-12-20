//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>
#include <stdexcept>

	
namespace my_opencv {

void dilation(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void erosion(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void opening(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void closing(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void gradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void hit_and_miss();  // Not yet implemented.
void top_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);
void bottom_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations);

}

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void morphological_operation()
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

	const std::string windowName1("morphological operation - original");
	const std::string windowName2("morphological operation - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
			//cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
		const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1));
		//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1));

		const int iterations = 1;
		cv::Mat result;
		//my_opencv::dilation(gray, result, selement, iterations);
		my_opencv::erosion(gray, result, selement, iterations);
		//my_opencv::opening(gray, result, selement, iterations);
		//my_opencv::closing(gray, result, selement, iterations);
		//my_opencv::gradient(gray, result, selement, iterations);
		//my_opencv::hit_and_miss();
		//my_opencv::top_hat(gray, result, selement, iterations);
		//my_opencv::bottom_hat(gray, result, selement, iterations);

		//
		cv::imshow(windowName1, img);
		cv::imshow(windowName2, result);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace my_opencv
