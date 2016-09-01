//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <iostream>
#include <ctime>


namespace {
namespace local {

// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
const int N = 4;

// ring image buffer
IplImage **buf = 0;
int last = 0;

// temporary images
IplImage *mhi = 0;  // MHI
IplImage *orient = 0;  // orientation
IplImage *mask = 0;  // valid orientation mask
IplImage *segmask = 0;  // motion segmentation map
CvMemStorage *storage = 0;  // temporary storage

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
void  update_mhi(IplImage* img, IplImage* dst, int diff_threshold)
{
	double timestamp = (double)clock() / CLOCKS_PER_SEC;  // get current time in seconds
	CvSize size = cvSize(img->width, img->height);  // get current frame size
	int i, idx1 = last, idx2;
	IplImage *silh;
	CvSeq *seq;
	CvRect comp_rect;
	double count;
	double angle;
	CvPoint center;
	double magnitude;
	CvScalar color;

	// allocate images at the beginning or reallocate them if the frame size is changed
	if (!mhi || mhi->width != size.width || mhi->height != size.height)
	{
		if (buf == 0)
		{
			buf = (IplImage**)malloc(N * sizeof(buf[0]));
			memset( buf, 0, N * sizeof(buf[0]));
		}

		for (i = 0; i < N; ++i)
		{
			cvReleaseImage(&buf[i]);
			buf[i] = cvCreateImage(size, IPL_DEPTH_8U, 1);
			cvZero(buf[i]);
		}
		cvReleaseImage(&mhi);
		cvReleaseImage(&orient);
		cvReleaseImage(&segmask);
		cvReleaseImage(&mask);

		mhi = cvCreateImage(size, IPL_DEPTH_32F, 1);
		cvZero(mhi);  // clear MHI at the beginning
		orient = cvCreateImage(size, IPL_DEPTH_32F, 1);
		segmask = cvCreateImage(size, IPL_DEPTH_32F, 1);
		mask = cvCreateImage(size, IPL_DEPTH_8U, 1);
	}

	cvCvtColor(img, buf[last], CV_BGR2GRAY);  // convert frame to grayscale

	idx2 = (last + 1) % N;  // index of (last - (N-1))th frame
	last = idx2;

	silh = buf[idx2];
	cvAbsDiff(buf[idx1], buf[idx2], silh);  // get difference between frames

	cvThreshold(silh, silh, diff_threshold, 1, CV_THRESH_BINARY);  // threshold it
	cvUpdateMotionHistory(silh, mhi, timestamp, MHI_DURATION);  // update MHI

	// convert MHI to blue 8u image
	cvCvtScale(mhi, mask, 255. / MHI_DURATION, (MHI_DURATION - timestamp) * 255. / MHI_DURATION);
	cvZero(dst);
	cvMerge(mask, NULL, NULL, NULL, dst);

	// calculate motion gradient orientation and valid orientation mask
	cvCalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	if (!storage)
		storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(storage);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. it is not used further
	seq = cvSegmentMotion(mhi, segmask, storage, timestamp, MAX_TIME_DELTA);

	// iterate through the motion components,
	// one more iteration (i == -1) corresponds to the whole image (global motion)
	for (i = -1; i < seq->total; ++i)
	{
		if (i < 0)  // case of the whole image
		{
			comp_rect = cvRect(0, 0, size.width, size.height);
			color = CV_RGB(255, 255, 255);
			magnitude = 100;
		}
		else  // i-th motion component
		{
			comp_rect = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;
			if (comp_rect.width + comp_rect.height < 100)  // reject very small components
				continue;
			color = CV_RGB(255, 0, 0);
			magnitude = 30;
		}

		// select component ROI
		cvSetImageROI(silh, comp_rect);
		cvSetImageROI(mhi, comp_rect);
		cvSetImageROI(orient, comp_rect);
		cvSetImageROI(mask, comp_rect);

		// calculate orientation
		angle = cvCalcGlobalOrientation(orient, mask, mhi, timestamp, MHI_DURATION);
		angle = 360.0 - angle;  // adjust for images with top-left origin

		count = cvNorm(silh, NULL, CV_L1, NULL);  // calculate number of points within silhouette ROI

		cvResetImageROI(mhi);
		cvResetImageROI(orient);
		cvResetImageROI(mask);
		cvResetImageROI(silh);

		// check for the case of little motion
		if (count < comp_rect.width * comp_rect.height * 0.05)
			continue;

		// draw a clock with arrow indicating the direction
		center = cvPoint((comp_rect.x + comp_rect.width / 2), (comp_rect.y + comp_rect.height / 2));

		cvCircle(dst, center, cvRound(magnitude * 1.2), color, 3, CV_AA, 0);
		cvLine(
			dst, center,
			cvPoint(cvRound(center.x + magnitude * cos(angle * CV_PI / 180)), cvRound(center.y - magnitude * sin(angle * CV_PI / 180))),
			color, 3, CV_AA, 0
		);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void motion_history_image()
{
#if 1
	const int imageWidth = 640, imageHeight = 480;

	const int camId = -1;
	//CvCapture *capture = cvCaptureFromCAM(camId);
	CvCapture *capture = cvCreateCameraCapture(camId);
/*
	const double propPosMsec = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_MSEC);
	const double propPosFrames = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES);
	const double propPosAviRatio = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO);
	const double propFrameWidth = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	const double propFrameHeight = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
	const double propFps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	const double propFourCC = cvGetCaptureProperty(capture, CV_CAP_PROP_FOURCC);
	const double propFrameCount = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
	const double propFormat = cvGetCaptureProperty(capture, CV_CAP_PROP_FORMAT);
	const double propMode = cvGetCaptureProperty(capture, CV_CAP_PROP_MODE);
	const double propBrightness = cvGetCaptureProperty(capture, CV_CAP_PROP_BRIGHTNESS);
	const double propContrast = cvGetCaptureProperty(capture, CV_CAP_PROP_CONTRAST);
	const double propSaturation = cvGetCaptureProperty(capture, CV_CAP_PROP_SATURATION);
	const double propHue = cvGetCaptureProperty(capture, CV_CAP_PROP_HUE);
	const double propGain = cvGetCaptureProperty(capture, CV_CAP_PROP_GAIN);
	const double propExposure = cvGetCaptureProperty(capture, CV_CAP_PROP_EXPOSURE);
	const double propConvertRGB = cvGetCaptureProperty(capture, CV_CAP_PROP_CONVERT_RGB);
	const double propWhiteBalance = cvGetCaptureProperty(capture, CV_CAP_PROP_WHITE_BALANCE);
	const double propRectification = cvGetCaptureProperty(capture, CV_CAP_PROP_RECTIFICATION);
	const double propMonochrome = cvGetCaptureProperty(capture, CV_CAP_PROP_MONOCROME);

	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_MSEC, propPosMsec);
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, propPosFrames);
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_AVI_RATIO, propPosAviRatio);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, propFrameWidth);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, propFrameHeight);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, propFps);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FOURCC, propFourCC);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT, propFrameCount);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FORMAT, propFormat);
	cvSetCaptureProperty(capture, CV_CAP_PROP_MODE, propMode);
	cvSetCaptureProperty(capture, CV_CAP_PROP_BRIGHTNESS, propBrightness);
	cvSetCaptureProperty(capture, CV_CAP_PROP_CONTRAST, propContrast);
	cvSetCaptureProperty(capture, CV_CAP_PROP_SATURATION, propSaturation);
	cvSetCaptureProperty(capture, CV_CAP_PROP_HUE, propHue);
	cvSetCaptureProperty(capture, CV_CAP_PROP_GAIN, propGain);
	cvSetCaptureProperty(capture, CV_CAP_PROP_EXPOSURE, propExposure);
	cvSetCaptureProperty(capture, CV_CAP_PROP_CONVERT_RGB, propConvertRGB);
	cvSetCaptureProperty(capture, CV_CAP_PROP_WHITE_BALANCE, propWhiteBalance);
	cvSetCaptureProperty(capture, CV_CAP_PROP_RECTIFICATION, propRectification);
	cvSetCaptureProperty(capture, CV_CAP_PROP_MONOCROME, propMonochrome);
*/
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imageHeight);
#else
	//const std::string avi_filename("./data/machine_vision/opencv/flycap-0001.avi");
	const std::string avi_filename("./data/machine_vision/opencv/tree.avi");
	//CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());
	CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
#endif

	if (capture)
	{
		IplImage *frame = NULL;
		IplImage *image = NULL;
		IplImage *motion = NULL;

		const char *windowName = "motion histroy image";
		cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

		for (;;)
		{
			frame = cvQueryFrame(capture);

			if (NULL == frame) continue;

#if 1
			if (NULL == image) image = cvCloneImage(frame);
			else cvCopy(frame, image);
#else
			image = frame;
#endif
			if (NULL == image) continue;

#if 1
			if (image->origin != IPL_ORIGIN_TL)
				cvFlip(image, image, 0);  // flip vertically (around x-axis)
#else
			if (image->origin != IPL_ORIGIN_TL)
				cvFlip(image, image, -1);  // flip vertically & horizontally (around both axes)
			else
				cvFlip(image, image, 1);  // flip horizontally (around y-axis)
#endif

			if (!motion)
			{
				motion = cvCloneImage(frame);
				cvZero(motion);
			}

			local::update_mhi(image, motion, 30);
			cvShowImage(windowName, motion);

			if (cvWaitKey(1) >= 0)
				break;
		}

		cvReleaseImage(&image);
		cvReleaseImage(&motion);

		cvReleaseCapture(&capture);
		cvDestroyWindow(windowName);
	}
}

}  // namespace my_opencv
