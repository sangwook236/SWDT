//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <cassert>

#define _INITIALIZE_HISTOGRAM_FROM_FILE 1


namespace {
namespace local {

const int HISTO_BIN_COUNT = 16;
const int MIN_HUE = 0, MAX_HUE = 180;

class ObjectTracker
{
public:
	ObjectTracker()
	: hsvImage_(NULL), hueImage_(NULL), maskImage_(NULL), backProjImage_(NULL), histo_(NULL)
	{
	}
	~ObjectTracker()
	{
	}

public:
	void initialize(const int width, const int height)
	{
		const CvSize imageSize = cvSize(width, height);

		hsvImage_ = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);
		hueImage_ = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
		maskImage_ = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
		backProjImage_ = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

		int histoBinCount1 = HISTO_BIN_COUNT;
		int histoBinCounts[] = { histoBinCount1 };
		float histoRange1[] = { MIN_HUE, MAX_HUE };
		float *histoRanges[] = { histoRange1 };
		histo_ = cvCreateHist(1, histoBinCounts, CV_HIST_ARRAY, histoRanges, 1);
	}

	void finalize()
	{
		cvReleaseHist(&histo_);

		cvReleaseImage(&backProjImage_);
		cvReleaseImage(&maskImage_);
		cvReleaseImage(&hueImage_);
		cvReleaseImage(&hsvImage_);
	}

	bool updateHistogram(const IplImage *image, const CvRect &targetRect, const int minHue, const int maxHue, const int minSaturation, const int maxSaturation, const int minValue, const int maxValue)
	{
		if (NULL == image) return false;
		if (NULL == hsvImage_ || NULL == hueImage_ || NULL == maskImage_) return false;

#if defined(__GNUC__)
		if (strcasecmp(image->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
		if (_stricmp(image->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(image, hsvImage_, CV_RGB2HSV);
#if defined(__GNUC__)
		else if (strcasecmp(image->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
		else if (_stricmp(image->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(image, hsvImage_, CV_BGR2HSV);
		else
		{
			std::cerr << "image cannot be converted !!!" << std::endl;
			return false;
		}

		//
		cvInRangeS(hsvImage_, cvScalar(minHue, minSaturation, minValue, 0), cvScalar(maxHue, maxSaturation, maxValue, 0), maskImage_);
		cvSplit(hsvImage_, hueImage_, NULL, NULL, NULL);

		//
		cvSetImageROI(hueImage_, targetRect);
		cvSetImageROI(maskImage_, targetRect);

		cvCalcHist(&hueImage_, histo_, 0, maskImage_);
		float maxHistoVal = 0.0f;
		cvGetMinMaxHistValue(histo_, NULL, &maxHistoVal, NULL, NULL);

		cvConvertScale(histo_->bins, histo_->bins, maxHistoVal ? 255.0 / maxHistoVal : 0.0, 0.0);

		cvResetImageROI(hueImage_);
		cvResetImageROI(maskImage_);

		return true;
	}

	bool updateHistogram(const IplImage *targetImage, const int minHue, const int maxHue, const int minSaturation, const int maxSaturation, const int minValue, const int maxValue)
	{
		if (NULL == targetImage) return false;

		const CvSize imageSize = cvGetSize(targetImage);

		IplImage *hsv = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);
		IplImage *hue = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);
		IplImage *mask = cvCreateImage(imageSize, IPL_DEPTH_8U, 1);

#if defined(__GNUC__)
		if (strcasecmp(targetImage->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
		if (_stricmp(targetImage->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(targetImage, hsv, CV_RGB2HSV);
#if defined(__GNUC__)
		else if (strcasecmp(targetImage->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
		else if (_stricmp(targetImage->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(targetImage, hsv, CV_BGR2HSV);
		else
		{
			std::cerr << "targetImage cannot be converted !!!" << std::endl;
			return false;
		}

		//
		cvInRangeS(hsv, cvScalar(minHue, minSaturation, minValue, 0), cvScalar(maxHue, maxSaturation, maxValue, 0), mask);
		cvSplit(hsv, hue, NULL, NULL, NULL);

		//
		cvCalcHist(&hue, histo_, 0, mask);
		float maxHistoVal = 0.0f;
		cvGetMinMaxHistValue(histo_, NULL, &maxHistoVal, NULL, NULL);

		cvConvertScale(histo_->bins, histo_->bins, maxHistoVal ? 255.0 / maxHistoVal : 0.0, 0.0);

		cvReleaseImage(&mask);
		cvReleaseImage(&hue);
		cvReleaseImage(&hsv);

		return true;
	}

	bool track(const IplImage *image, CvRect &searchWindow, CvBox2D &trackingBox, const int minHue, const int maxHue, const int minSaturation, const int maxSaturation, const int minValue, const int maxValue)
	{
		if (NULL == image) return false;
		if (NULL == hsvImage_ || NULL == hueImage_ || NULL == maskImage_ || NULL == backProjImage_) return false;

#if defined(__GNUC__)
		if (strcasecmp(image->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
		if (_stricmp(image->channelSeq, "RGB") == 0)
#endif
			cvCvtColor(image, hsvImage_, CV_RGB2HSV);
#if defined(__GNUC__)
		else if (strcasecmp(image->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
		else if (_stricmp(image->channelSeq, "BGR") == 0)
#endif
			cvCvtColor(image, hsvImage_, CV_BGR2HSV);
		else
		{
			std::cerr << "image cannot be converted !!!" << std::endl;
			return false;
		}

		//
		cvInRangeS(hsvImage_, cvScalar(minHue, minSaturation, minValue, 0), cvScalar(maxHue, maxSaturation, maxValue, 0), maskImage_);
		cvSplit(hsvImage_, hueImage_, NULL, NULL, NULL);

		//
		cvCalcBackProject(&hueImage_, backProjImage_, histo_);
		cvAnd(backProjImage_, maskImage_, backProjImage_, NULL);

		CvConnectedComp trackingComp;
		cvCamShift(
			backProjImage_, searchWindow,
			cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
			&trackingComp, &trackingBox
		);

		searchWindow = trackingComp.rect;

		return true;
	}

	IplImage * getBackProjectionImage()  {  return backProjImage_;  }
	const IplImage * getBackProjectionImage() const  {  return backProjImage_;  }

	CvHistogram * getHistogram()  {  return histo_;  }
	const CvHistogram * getHistogram() const  {  return histo_;  }

private:
	IplImage *hsvImage_;
	IplImage *hueImage_;
	IplImage *maskImage_;
	IplImage *backProjImage_;

	CvHistogram *histo_;
};

IplImage *image = NULL;
bool isSelectingTargetObject = false;
bool isTrackingObject = false;
bool isHistogramUpdated = false;
CvPoint startPt;
CvRect selection;

CvScalar hsv2rgb(float hue)
{
	int rgb[3], p, sector;
	static const int sector_data[][3] = { {0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2} };
	hue *= 0.033333333333333333333333333333333f;
	sector = cvFloor(hue);
	p = cvRound(255 * (hue - sector));
	p ^= sector & 1 ? 255 : 0;

	rgb[sector_data[sector][0]] = 255;
	rgb[sector_data[sector][1]] = 0;
	rgb[sector_data[sector][2]] = p;

	return cvScalar(rgb[2], rgb[1], rgb[0], 0);
}

void drawHistogram(const CvHistogram *histo, IplImage *histoImage)
{
	cvZero(histoImage);

	if (NULL == histo) return;

	const int binWidth = histoImage->width / HISTO_BIN_COUNT;
	for (int i = 0; i < HISTO_BIN_COUNT; ++i)
	{
		const int val = cvRound(cvGetReal1D(histo->bins, i) * histoImage->height / 255);
		const CvScalar color = hsv2rgb(i * 180.0f / HISTO_BIN_COUNT);

		cvRectangle(
			histoImage,
			cvPoint(i * binWidth, histoImage->height), cvPoint((i+1) * binWidth, histoImage->height - val),
			color, -1, 8, 0
		);
	}
}

void onMouseHandler(int event, int x, int y, int flags, void *param)
{
	if (NULL == image) return;

	if (image->origin)  // bottom-left origin
		y = image->height - y;

	if (isSelectingTargetObject)
	{
		selection.x = MIN(x, startPt.x);
		selection.y = MIN(y, startPt.y);
		selection.width = selection.x + CV_IABS(x - startPt.x);
		selection.height = selection.y + CV_IABS(y - startPt.y);

		selection.x = MAX(selection.x, 0);
		selection.y = MAX(selection.y, 0);
		selection.width = MIN(selection.width, image->width);
		selection.height = MIN(selection.height, image->height);
		selection.width -= selection.x;
		selection.height -= selection.y;
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		startPt = cvPoint(x, y);
		selection = cvRect(x, y, 0, 0);
		isSelectingTargetObject = true;
		break;

	case CV_EVENT_LBUTTONUP:
		isSelectingTargetObject = false;
		if (selection.width > 0 && selection.height > 0)
		{
			isHistogramUpdated = true;
			isTrackingObject = true;
		}
		break;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void object_tracking()
{
	const std::string avi_filename("./machine_vision_data/opencv/osp_test.wmv");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());

	if (!capture)
	{
		std::cerr << "could not initialize capturing..." << std::endl;
		return;
	}

	const std::string usage(
		"Hot keys: \n"
		"\tESC - quit the program\n"
		"\tc - stop the tracking\n"
		"\tb - switch to/from backprojection view\n"
		"\th - show/hide object histogram\n"
		"To initialize tracking, select the object with mouse"
	);
	std::cout << usage << std::endl;

	//
	local::ObjectTracker objectTracker;

	//
	const char *windowName = "object tracking by camshift";
	const char *histoWindowName = "histogram for object tracking by camshift";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvNamedWindow(histoWindowName, CV_WINDOW_AUTOSIZE);

	cvSetMouseCallback(windowName, local::onMouseHandler, NULL);

	int minHue = local::MIN_HUE, maxHue = local::MAX_HUE, minSaturation = 30, maxSaturation = 256, minValue = 10, maxValue = 256;
	cvCreateTrackbar("Min Saturation in HSV", windowName, &minSaturation, 256, NULL);
	cvCreateTrackbar("Mix Value in HSV", windowName, &minValue, 256, NULL);
	cvCreateTrackbar("Max Value in HSV", windowName, &maxValue, 256, NULL);

	//
	const int histoImageWidth = 320, histoImageHeight = 200;
	IplImage *histoImage = cvCreateImage(cvSize(histoImageWidth, histoImageHeight), IPL_DEPTH_8U, 3);

	cvZero(histoImage);

	//
	bool isInitialized = false;
	bool isHistrogramShown = true;
	bool isBackProjectionMode = false;
	CvRect searchWindow;
	CvBox2D trackingBox;
	while (true)
	{
		IplImage *frame = cvQueryFrame(capture);
		if (!frame) break;

		if (!isInitialized)
		{
			local::image = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
			local::image->origin = frame->origin;  // if 0, top-left origin. if 1, bottom-left origin.

			objectTracker.initialize(frame->width, frame->height);

#if defined(_INITIALIZE_HISTOGRAM_FROM_FILE)
			const std::string targetImageFileName("./machine_vision_data/opencv/target_osp_robot.png");
			IplImage *targetImage = cvLoadImage(targetImageFileName.c_str());

			objectTracker.updateHistogram(targetImage, minHue, maxHue, minSaturation, maxSaturation, MIN(minValue, maxValue), MAX(minValue, maxValue));
			local::drawHistogram(objectTracker.getHistogram(), histoImage);

			cvReleaseImage(&targetImage);

			local::isTrackingObject = true;
			// TODO [modify] >>
			searchWindow = cvRect(0, 0, frame->width, frame->height);
			//searchWindow = cvRect(335, 200, 70, 30);
#endif

			isInitialized = true;
		}

		cvCopy(frame, local::image, NULL);

		if (local::isTrackingObject)
		{
#if defined(_INITIALIZE_HISTOGRAM_FROM_FILE)
#else
			if (isHistogramUpdated)
			{
				searchWindow = selection;
				objectTracker.updateHistogram(local::image, searchWindow, minHue, maxHue, minSaturation, maxSaturation, MIN(minValue, maxValue), MAX(minValue, maxValue));
				drawHistogram(objectTracker.getHistogram(), histoImage);
			}
#endif

			objectTracker.track(local::image, searchWindow, trackingBox, minHue, maxHue, minSaturation, maxSaturation, MIN(minValue, maxValue), MAX(minValue, maxValue));

#if !defined(_INITIALIZE_HISTOGRAM_FROM_FILE)
			if (isHistogramUpdated) isHistogramUpdated = false;
#endif
			//
			if (isBackProjectionMode)
				cvCvtColor(objectTracker.getBackProjectionImage(), local::image, CV_GRAY2BGR);

			if (!local::image->origin)  // top-left origin
				trackingBox.angle = -trackingBox.angle;

			cvEllipseBox(local::image, trackingBox, CV_RGB(255,0,0), 3, CV_AA, 0);
		}

#if !defined(_INITIALIZE_HISTOGRAM_FROM_FILE)
		if (isSelectingTargetObject && selection.width > 0 && selection.height > 0)
		{
			cvSetImageROI(local::image, selection);
			cvXorS(local::image, cvScalarAll(255), local::image, 0);
			cvResetImageROI(local::image);
		}
#endif

		cvShowImage(windowName, local::image);
		if (isHistrogramShown) cvShowImage(histoWindowName, histoImage);

		//
		const int c = cvWaitKey(10);
		if (27 == c)  // ESC
			break;
		switch (c)
		{
		case 'b':
			isBackProjectionMode = !isBackProjectionMode;
			break;
		case 'c':
			local::isTrackingObject = false;
			cvZero(histoImage);
			break;
		case 'h':
			isHistrogramShown = !isHistrogramShown;
			if (isHistrogramShown)
				cvNamedWindow(histoWindowName, CV_WINDOW_AUTOSIZE);
			else
				cvDestroyWindow(histoWindowName);
			break;
		default:
			break;
		}
	}

	//
	objectTracker.finalize();

	cvReleaseImage(&histoImage);

	cvReleaseCapture(&capture);
	cvDestroyWindow(windowName);
	cvDestroyWindow(histoWindowName);
}

}  // namespace my_opencv
