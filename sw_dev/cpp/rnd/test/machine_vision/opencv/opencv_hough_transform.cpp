//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <cassert>


namespace {
namespace local {

void hough_line_standard(IplImage *srcImage, IplImage *grayImage)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvCanny(grayImage, grayImage, 50.0, 200.0, 3);

	const CvSeq *lines = cvHoughLines2(
		grayImage,
		storage,
		CV_HOUGH_STANDARD,
		1.0,
		CV_PI / 180.0,
		100,
		0,
		0
	);

	//const int lineCount = lines->total;
	const int lineCount = MIN(lines->total, 100);
	for (int i = 0; i < lineCount; ++i)
	{
		const float *line = (float *)cvGetSeqElem(lines, i);
		const float &rho = line[0];  // a distance between (0,0) point and the line,
		const float &theta = line[1];  // the angle between x-axis and the normal to the line

		const double a = std::cos(theta), b = std::sin(theta);
		const double x0 = a * rho, y0 = b * rho;

		CvPoint pt1, pt2;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cvLine(srcImage, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA, 0);
	}

	cvClearMemStorage(storage);
}

void hough_line_probabilistic(IplImage *srcImage, IplImage *grayImage)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvCanny(grayImage, grayImage, 50.0, 200.0, 3);

	const CvSeq *lines = cvHoughLines2(
		grayImage,
		storage,
		CV_HOUGH_PROBABILISTIC,
		1.0,
		CV_PI / 180.0,
		80,
		30.0,
		10.0
	);

	//const int lineCount = lines->total;
	const int lineCount = MIN(lines->total, 100);
	for (int i = 0; i < lineCount; ++i)
	{
		const CvPoint *line = (CvPoint *)cvGetSeqElem(lines, i);
		cvLine(srcImage, line[0], line[1], CV_RGB(255,0,0), 1, CV_AA, 0);
	}

	cvClearMemStorage(storage);
}

void hough_circle(IplImage *srcImage, IplImage *grayImage)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	cvSmooth(grayImage, grayImage, CV_GAUSSIAN, 9, 9, 0.0, 0.0);

	const CvSeq *circles = cvHoughCircles(
		grayImage,
		storage,
		CV_HOUGH_GRADIENT,
		2,
		grayImage->height / 4.0,
		200,
		100,
		0,
		0
	);

	//const int circleCount = circles->total;
	const int circleCount = MIN(circles->total, 100);
	for (int i = 0; i < circleCount; ++i)
	{
		const float *circle = (float *)cvGetSeqElem(circles, i);
		const float &x = circle[0];
		const float &y = circle[1];
		const float &r = circle[2];
		cvCircle(srcImage, cvPoint(cvRound(x), cvRound(y)), 3, CV_RGB(0,255,0), CV_FILLED, CV_AA, 0);
		cvCircle(srcImage, cvPoint(cvRound(x), cvRound(y)), cvRound(r), CV_RGB(255,0,0), 1, CV_AA, 0);
	}

	cvClearMemStorage(storage);
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void hough_transform()
{
	const std::string imageFileName("./data/machine_vision/opencv/hough_line.png");

	const char *windowName = "hough transform";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

	//
	IplImage *srcImage = cvLoadImage(imageFileName.c_str(), 1);

	IplImage *grayImage = NULL;
	if (1 == srcImage->nChannels)
		cvCopy(srcImage, grayImage, NULL);
	else
	{
		grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
#if defined(__GNUC__)
		if (strcasecmp(srcImage->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
		if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
		else if (strcasecmp(srcImage->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
		else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
		else
			assert(false);
		grayImage->origin = srcImage->origin;
	}

	//
	//local::hough_line_standard(srcImage, grayImage);
	//local::hough_line_probabilistic(srcImage, grayImage);
	local::hough_circle(srcImage, grayImage);

	//
	cvShowImage(windowName, srcImage);
	cvWaitKey();

	//
	cvReleaseImage(&grayImage);
	cvReleaseImage(&srcImage);

	cvDestroyWindow(windowName);
}

}  // namespace my_opencv
