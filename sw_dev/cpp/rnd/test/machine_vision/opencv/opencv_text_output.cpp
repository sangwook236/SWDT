//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>


namespace {
namespace local {

void text_output_1()
{
	const int imageWidth = 1280, imageHeight = 480;

	//
	IplImage *image = cvCreateImage(cvSize(imageWidth, imageHeight), IPL_DEPTH_8U, 1);
	cvSet(image, CV_RGB(0, 0, 0));

	//
	CvFont font;

	//const int fontFace = CV_FONT_HERSHEY_SIMPLEX;
	//const int fontFace = CV_FONT_HERSHEY_PLAIN;
	//const int fontFace = CV_FONT_HERSHEY_DUPLEX;
	const int fontFace = CV_FONT_HERSHEY_COMPLEX;
	//const int fontFace = CV_FONT_HERSHEY_TRIPLEX;
	//const int fontFace = CV_FONT_HERSHEY_COMPLEX_SMALL;
	//const int fontFace = CV_FONT_HERSHEY_SCRIPT_SIMPLEX;
	//const int fontFace = CV_FONT_HERSHEY_SCRIPT_COMPLEX;

	cvInitFont(&font, fontFace, 3.0f /*hscale*/, 3.0f /*vscale*/, 0.0, 1);

	cvPutText(image, "Hello, OpenCV !!!", cvPoint(10, 200), &font, CV_RGB(255, 255, 255));

	//
	const char *windowName = "text output window";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName, imageWidth, imageHeight);

	cvShowImage(windowName, image);
	cvWaitKey();

	cvReleaseImage(&image);

	cvDestroyWindow(windowName);
}

void text_output_2()
{
	const int imageWidth = 1280, imageHeight = 480;

	//
	cv::Mat image(cv::Size(imageWidth, imageHeight), CV_8UC1);
	image.setTo(cv::Scalar(0, 0, 0));

	//const int fontFace = cv::_FONT_HERSHEY_SIMPLEX;
	//const int fontFace = cv::FONT_HERSHEY_PLAIN;
	//const int fontFace = cv::FONT_HERSHEY_DUPLEX;
	const int fontFace = cv::FONT_HERSHEY_COMPLEX;
	//const int fontFace = CV_FONT_HERSHEY_TRIPLEX;
	//const int fontFace = CV_FONT_HERSHEY_COMPLEX_SMALL;
	//const int fontFace = CV_FONT_HERSHEY_SCRIPT_SIMPLEX;
	//const int fontFace = CV_FONT_HERSHEY_SCRIPT_COMPLEX;

	const double fontScale = 0.5;
	cv::putText(image, "Hello, OpenCV !!!", cv::Point(10, 200), fontFace, fontScale, CV_RGB(255, 0, 255), 1, 8, false);

	//
	const std::string windowName("Text output window");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName, image);
	cv::waitKey();

	cv::destroyWindow(windowName);
}

void text_output_3()
{
	const std::string text("Funny text inside the box");

	const int fontFace = cv::FONT_HERSHEY_TRIPLEX;
	const double fontScale = 2.0;
	const int thickness = 3;
	const bool drawBox = true;

	cv::Mat img(600, 800, CV_8UC3, cv::Scalar::all(0));

	int baseline = 0;
	const cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	// Center the text.
	const cv::Point textOrg((img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);

	if (drawBox)
	{
		// Draw the box.
		cv::rectangle(img, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height), cv::Scalar(0, 0, 255));
		// And the baseline first.
		cv::line(img, textOrg + cv::Point(0, thickness), textOrg + cv::Point(textSize.width, thickness), cv::Scalar(0, 0, 255));
	}

	// Then put the text itself.
	cv::putText(img, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, cv::LINE_8);

	// Display.
	const std::string windowName("text output window");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	cv::imshow(windowName, img);
	cv::waitKey();

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void text_output()
{
	//local::text_output_1();
	//local::text_output_2();
	local::text_output_3();
}

}  // namespace my_opencv
