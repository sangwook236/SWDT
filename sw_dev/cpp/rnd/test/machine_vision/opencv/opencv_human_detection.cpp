//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <list>
#include <string>


namespace {
namespace local {

void human_detection_using_hog()
{
	std::list<std::string> filenames;

	filenames.push_back("./data/machine_vision/opencv/human_01.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_02.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_03.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_04.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_05.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_06.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_07.jpg");
	filenames.push_back("./data/machine_vision/opencv/human_08.jpg");

	const std::string windowName("human detection");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	//
	const double hitThreshold = 0;
	const cv::Size winStride(8, 8);
	const cv::Size padding(32, 32);
	const double scale = 1.05;
	const int groupThreshold = 2;

	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {
		cv::Mat img(cv::imread(*it, CV_LOAD_IMAGE_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		//
		std::vector<cv::Rect> found;
		const double t = (double)cv::getTickCount();
		// run the detector with default parameters.
		// to get a higher hit-rate (and more false alarms, respectively), decrease the hitThreshold and groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
		hog.detectMultiScale(img, found, hitThreshold, winStride, padding, scale, groupThreshold);
		const double et = ((double)cv::getTickCount() - t) * 1000.0 / cv::getTickFrequency();
		std::cout << "detection time = " << et << "ms" << std::endl;

		std::vector<cv::Rect> found_filtered;
		found_filtered.reserve(found.size());
		for (std::vector<cv::Rect>::const_iterator cit = found.begin(); cit != found.end(); ++cit)
		{
			std::vector<cv::Rect>::const_iterator cit2 = found.begin();
			for (; cit2 != found.end(); ++cit2)
				if (cit2 != cit && (*cit & *cit2) == *cit)
					break;
			if (found.end() == cit2)
				found_filtered.push_back(*cit);
		}
		for (std::vector<cv::Rect>::iterator it = found.begin(); it != found.end(); ++it)
		{
			// the HOG detector returns slightly larger rectangles than the real objects.
			// so we slightly shrink the rectangles to get a nicer output.
			it->x += cvRound(it->width * 0.1);
			it->width = cvRound(it->width * 0.8);
			it->y += cvRound(it->height * 0.07);
			it->height = cvRound(it->height * 0.8);
			cv::rectangle(img, it->tl(), it->br(), CV_RGB(0,255,0), 3);
		}

		cv::imshow(windowName, img);

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyWindow(windowName);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void human_detection()
{
	local::human_detection_using_hog();
}

}  // namespace my_opencv
