//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


namespace {
namespace local {

void normalized_cross_correlation()
{
	const std::string filename1("machine_vision_data\\opencv\\melon_target.png");
	const std::string filename2("machine_vision_data\\opencv\\melon_1.png");
	//const std::string filename2("machine_vision_data\\opencv\\melon_2.png");
	//const std::string filename2("machine_vision_data\\opencv\\melon_3.png");

	const cv::Mat &templ0 = cv::imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image(cv::imread(filename2, CV_LOAD_IMAGE_GRAYSCALE));

	const int TEMPL_WIDTH = 120;
	const int TEMPL_HEIGHT = 120;

	if (image.rows <= TEMPL_HEIGHT || image.cols <= TEMPL_WIDTH)
	{
		std::cout << "image is too small" << std::endl;
		return;
	}

	cv::Mat templ;
	cv::resize(templ0, templ, cv::Size(TEMPL_WIDTH, TEMPL_HEIGHT), 0.0, 0.0, cv::INTER_LINEAR);

	const size_t pos_x = std::rand() % (image.cols - TEMPL_WIDTH);
	const size_t pos_y = std::rand() % (image.rows - TEMPL_HEIGHT);
	std::cout << "\tinserted location: (" << pos_x << ", " << pos_y << ")" << std::endl;

	// insert template to image
#if defined(__GNUC__)
    {
        cv::Mat image_roi(image, cv::Range(pos_y, pos_y + TEMPL_HEIGHT), cv::Range(pos_x, pos_x + TEMPL_WIDTH));
       	templ.copyTo(image_roi);
    }
#else
	templ.copyTo(image(cv::Range(pos_y, pos_y + TEMPL_HEIGHT), cv::Range(pos_x, pos_x + TEMPL_WIDTH)));
#endif

	// perform NCC
	// CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_TM_CCORR, CV_TM_CCORR_NORMED, CV_TM_CCOEFF, CV_TM_CCOEFF_NORMED
	const int comparison_method = CV_TM_CCOEFF_NORMED;
	cv::Mat result;  // a single-channel 32-bit floating-point

	const double &startTime = (double)cv::getTickCount();
	cv::matchTemplate(image, templ, result, comparison_method);
	const double &endTime = (double)cv::getTickCount();

	double minVal = 0.0, maxVal = 0.0;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	cv::Point matched_loc(-1, -1);
	switch (comparison_method)
	{
	// sum of squared differences
	case CV_TM_SQDIFF:
	case CV_TM_SQDIFF_NORMED:
		std::cout << "\tmin value: " << minVal << std::endl;
		matched_loc = minLoc;
		break;
	// correlation
	case CV_TM_CCORR:
	case CV_TM_CCORR_NORMED:
	// correlation coefficients
	case CV_TM_CCOEFF:
	case CV_TM_CCOEFF_NORMED:
		std::cout << "\tmax value: " << maxVal << std::endl;
		matched_loc = maxLoc;
		break;
	}

	std::cout << "\tmatched location: (" << matched_loc.x << ", " << matched_loc.y << ")" << std::endl;
	cv::rectangle(image, matched_loc, matched_loc + cv::Point(TEMPL_WIDTH, TEMPL_HEIGHT), CV_RGB(255, 0, 0), 2, 8, 0);
	std::cout << "\tprocessing time: " << ((endTime - startTime) / ((double)cv::getTickFrequency() * 1000.0)) << " ms" << std::endl;

	//
	const std::string windowName1("template matching - image");
	const std::string windowName2("template matching - template");
	cv::namedWindow(windowName1, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(windowName2, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName1, image);
	cv::imshow(windowName2, templ);

	cv::waitKey(0);

	cv::destroyWindow(windowName1);
	cv::destroyWindow(windowName2);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void template_matching()
{
	local::normalized_cross_correlation();
}

}  // namespace my_opencv
