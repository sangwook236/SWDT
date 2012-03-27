//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <cassert>


namespace {

void text_output_1()
{
	const int imageWidth = 1280, imageHeight = 480;
	const char *windowName = "text output window";

	//
	IplImage *image = cvCreateImage(cvSize(imageWidth, imageHeight), IPL_DEPTH_8U, 1);
	cvSet(image, CV_RGB(0, 0, 0));

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName, imageWidth, imageHeight);

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

	cvShowImage(windowName, image);
	cvWaitKey();

	//
	cvReleaseImage(&image);

	cvDestroyWindow(windowName);
}

void text_output_2()
{
	const int imageWidth = 1280, imageHeight = 480;
	const std::string windowName("text output window");

	//
	cv::Mat image(cv::Size(imageWidth, imageHeight), CV_8UC1);
	image.setTo(cv::Scalar(0,0,0));

	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

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

	cv::imshow(windowName, image);
	cv::waitKey();

	cv::destroyWindow(windowName);
}

}  // unnamed namespace

void text_output()
{
	//text_output_1();
	text_output_2();
}
