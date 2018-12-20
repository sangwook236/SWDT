//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <list>


namespace my_opencv {

// REF [file] >> ${CPP_RND_HOME}/test/machine_vision/opencv/opencv_util.cpp
void canny(const cv::Mat &gray, const int lowerEdgeThreshold, const int upperEdgeThreshold, const bool useL2, cv::Mat &edge);
void sobel(const cv::Mat &gray, const double thresholdRatio, cv::Mat &gradient);

}  // namespace my_opencv

namespace {
namespace local {

void color_channel_extraction()
{
	std::list<std::string> filenames;
#if 0
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
#elif 1
	filenames.push_back("../data/machine_vision/opencv/hand_left_1.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_right_1.jpg");

	filenames.push_back("../data/machine_vision/opencv/hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_02.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_03.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_04.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_06.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_07.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_08.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_09.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_10.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_11.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_12.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_13.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_14.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_15.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_16.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_17.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_18.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_19.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_20.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_21.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_22.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_23.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_24.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_25.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_26.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_27.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_28.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_29.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_30.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_31.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_32.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_33.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_34.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_35.jpg");
	filenames.push_back("../data/machine_vision/opencv/hand_36.jpg");
#elif 0
	filenames.push_back("../data/machine_vision/opencv/simple_hand_01.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_02.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_03.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_04.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_05.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_06.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_07.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_08.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_09.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_10.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_11.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_12.jpg");
	filenames.push_back("../data/machine_vision/opencv/simple_hand_13.jpg");
#endif

	const std::string windowName1("color filtering - channel 1");
	const std::string windowName2("color filtering - channel 2");
	const std::string windowName3("color filtering - channel 3");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName3, cv::WINDOW_AUTOSIZE);

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		const cv::Mat &img = cv::imread(*it, cv::IMREAD_COLOR);
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		cv::Mat img2 = img;
		//cv::cvtColor(img, img2, cv::COLOR_BGR2GRAY);  // BGR <-> gray
		//cv::cvtColor(img, img2, cv::COLOR_BGR2XYZ);  // BGR <-> CIE XYZ
		//cv::cvtColor(img, img2, cv::COLOR_BGR2YCrCb);  // BGR <-> YCrCb JPEG
		//cv::cvtColor(img, img2, cv::COLOR_BGR2HSV);  // BGR <-> HSV
		//cv::cvtColor(img, img2, cv::COLOR_BGR2HLS);  // BGR <-> HLS
		//cv::cvtColor(img, img2, cv::COLOR_BGR2Lab);  // BGR <-> CIE L*a*b*
		//cv::cvtColor(img, img2, cv::COLOR_BGR2Luv);  // BGR <-> CIE L*u*v*

		std::vector<cv::Mat> filtered_imgs;
		cv::split(img2, filtered_imgs);

		std::vector<cv::Mat> filtered_imgs2(3);
		cv::equalizeHist(filtered_imgs[0], filtered_imgs2[0]);
		cv::equalizeHist(filtered_imgs[1], filtered_imgs2[1]);
		cv::equalizeHist(filtered_imgs[2], filtered_imgs2[2]);

		std::vector<cv::Mat> filtered_imgs3(3);
#if 0
		const double thresholdRatio = 0.15;
		my_opencv::sobel(filtered_imgs2[0], thresholdRatio, filtered_imgs3[0]);
		my_opencv::sobel(filtered_imgs2[1], thresholdRatio, filtered_imgs3[1]);
		my_opencv::sobel(filtered_imgs2[2], thresholdRatio, filtered_imgs3[2]);
#else
		const int lowerEdgeThreshold = 20, upperEdgeThreshold = 50;
		const bool useL2 = true;
		my_opencv::canny(filtered_imgs2[0], lowerEdgeThreshold, upperEdgeThreshold, useL2, filtered_imgs3[0]);
		my_opencv::canny(filtered_imgs2[1], lowerEdgeThreshold, upperEdgeThreshold, useL2, filtered_imgs3[1]);
		my_opencv::canny(filtered_imgs2[2], lowerEdgeThreshold, upperEdgeThreshold, useL2, filtered_imgs3[2]);
#endif

		cv::imshow(windowName1, filtered_imgs3[0]);
		cv::imshow(windowName2, filtered_imgs3[1]);
		cv::imshow(windowName3, filtered_imgs3[2]);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
	cv::destroyWindow(windowName3);
}

void color_based_tracking()
{
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	const bool b1 = capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	const bool b2 = capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	const std::string windowName1("color filtering - original");
	const std::string windowName2("color filtering - processed");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::Mat frame, img;
	for (;;)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			//break;
			continue;
		}

		frame.copyTo(img);

		cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
#if 0
		// slowest: > 0.016 [sec]
		std::vector<cv::Mat> filtered_imgs;
		cv::split(img, filtered_imgs);

		const double &startTime = (double)cv::getTickCount();
		for (int r = 0; r < img.rows; ++r)
			for (int c = 0; c < img.cols; ++c)
			{
				const unsigned char &blue = filtered_imgs[0].at<unsigned char>(r, c);
				const unsigned char &green = filtered_imgs[1].at<unsigned char>(r, c);
				const unsigned char &red = filtered_imgs[2].at<unsigned char>(r, c);

				if (blue <= 64 && green <= 64 && red > 128)
					mask.at<unsigned char>(r, c) = 1;
			}
		const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
#elif 0
		// > 0.004 [sec]
		const double &startTime = (double)cv::getTickCount();
		for (int r = 0; r < img.rows; ++r)
			for (int c = 0; c < img.cols; ++c)
			{
				const cv::Vec3b &bgr = img.at<cv::Vec3b>(r, c);

				if (bgr[0] <= 64 && bgr[1] <= 64 && bgr[2] > 128)
					mask.at<unsigned char>(r, c) = 1;
			}
		const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
#elif 0
		// fastest: > 0.0013 [sec]
		const unsigned char *row = NULL;
		const double &startTime = (double)cv::getTickCount();
		for (int r = 0; r < img.rows; ++r)
		{
			row = (unsigned char *)img.ptr(r);  // get r-th row
			for (int c = 0; c < img.cols; ++c, row += 3)
			{
				if (row[0] <= 64 && row[1] <= 64 && row[2] > 128)
					mask.at<unsigned char>(r, c) = 1;
			}
		}
		const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
#else
		// fastest: > 0.0013 [sec]
		const unsigned char *pixels = img.data;
		const double &startTime = (double)cv::getTickCount();
		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c, pixels += 3)
			{
				if (pixels[0] <= 64 && pixels[1] <= 64 && pixels[2] > 128)
					mask.at<unsigned char>(r, c) = 1;
			}
		}
		const double &elapsedTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
#endif

		std::cout << elapsedTime << std::endl;

		cv::imshow(windowName1, img);
		cv::Mat filtered_img;
		img.copyTo(filtered_img, mask);
		cv::imshow(windowName2, filtered_img);

		const unsigned char key = cv::waitKey(1);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void color_filtering()
{
	//local::color_channel_extraction();
	local::color_based_tracking();
}

}  // namespace my_opencv
