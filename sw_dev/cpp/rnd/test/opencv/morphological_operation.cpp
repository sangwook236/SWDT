#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>


namespace {

void dilation(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	cv::dilate(src, dst, selement, cv::Point(-1, -1), iterations);
}

void erosion(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	cv::erode(src, dst, selement, cv::Point(-1, -1), iterations);
}

void opening(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// opening = dilation -> erosion
	cv::morphologyEx(src, dst, cv::MORPH_OPEN, selement, cv::Point(-1, -1), iterations);
}

void closing(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// closing = erosion -> dilation
	cv::morphologyEx(src, dst, cv::MORPH_CLOSE, selement, cv::Point(-1, -1), iterations);
}

void gradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// gradient = dilation - erosion
	cv::morphologyEx(src, dst, cv::MORPH_GRADIENT, selement, cv::Point(-1, -1), iterations);
}

void hit_and_miss()
{
	throw std::runtime_error("Oops !!! not implemented");
}

void top_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// top_hat = src - opening
	cv::morphologyEx(src, dst, cv::MORPH_TOPHAT, selement, cv::Point(-1, -1), iterations);
}

void bottom_hat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &selement, const int iterations)
{
	// bottom_hat = closing - src
	cv::morphologyEx(src, dst, cv::MORPH_BLACKHAT, selement, cv::Point(-1, -1), iterations);
}

}

void morphological_operation()
{
	std::list<std::string> filenames;
	filenames.push_back("opencv_data\\pic1.png");
	filenames.push_back("opencv_data\\pic2.png");
	filenames.push_back("opencv_data\\pic3.png");
	filenames.push_back("opencv_data\\pic4.png");
	filenames.push_back("opencv_data\\pic5.png");
	filenames.push_back("opencv_data\\pic6.png");
	filenames.push_back("opencv_data\\stuff.jpg");
	filenames.push_back("opencv_data\\synthetic_face.png");
	filenames.push_back("opencv_data\\puzzle.png");
	filenames.push_back("opencv_data\\fruits.jpg");
	filenames.push_back("opencv_data\\lena_rgb.bmp");
	filenames.push_back("opencv_data\\hand_01.jpg");
	filenames.push_back("opencv_data\\hand_05.jpg");
	filenames.push_back("opencv_data\\hand_24.jpg");

	const std::string windowName1("morphological operation - original");
	const std::string windowName2("morphological operation - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat img = cv::imread(*it, CV_LOAD_IMAGE_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat gray;
		if (1 == img.channels())
			img.copyTo(gray);
		else
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			//cv::cvtColor(img, gray, CV_RGB2GRAY);

		//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)); 
		const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5), cv::Point(-1, -1)); 
		//const cv::Mat &selement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1)); 

		const int iterations = 1;
		cv::Mat result;
		//dilation(gray, result, selement, iterations);
		erosion(gray, result, selement, iterations);
		//opening(gray, result, selement, iterations);
		//closing(gray, result, selement, iterations);
		//gradient(gray, result, selement, iterations);
		//hit_and_miss();
		//top_hat(gray, result, selement, iterations);
		//bottom_hat(gray, result, selement, iterations);

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
