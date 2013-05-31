//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#if 0
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include <opencv/cvcam.h>
#else
#include <opencv2/opencv.hpp>
#endif
#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <stdexcept>


namespace {
namespace local {

bool isCapturing = true, isThreadTerminated = false;

void capture_image_from_file()
{
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 176, imageHeight = 144;

	const std::string avi_filename("./machine_vision_data/opencv/flycap-0001.avi");
	const std::string windowName("capturing from file");

#if 0
	//
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());

	cvNamedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName.c_str(), imageWidth, imageHeight);

	std::cout << "press any key if want to finish ..." << std::endl;
	IplImage *frame = NULL;
	while (cvWaitKey(1) < 0)
	{
		//cvGrabFrame(capture);
		//frame = cvRetrieveFrame(capture);
		frame = cvQueryFrame(capture);

		cvShowImage(windowName, frame);
	}
	std::cout << "end capturing ..." << std::endl;

	//
	cvReleaseCapture(&capture);
	cvDestroyWindow(windowName);
#else
	cv::VideoCapture capture(avi_filename);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cv::resizeWindow(windowName, imageWidth, imageHeight);

	std::cout << "press any key if want to finish ..." << std::endl;
	cv::Mat frame;
	while (cv::waitKey(1) < 0)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		cv::imshow(windowName, frame);
	}
	std::cout << "end capturing ..." << std::endl;

	cv::destroyWindow(windowName);
#endif
}

#if 0  // OpenCV 1.0
void opencv_capture_callback(IplImage *image);
DWORD WINAPI opencv_capture_thread_proc(LPVOID param);

void capture_image_from_cam()
{
	const int camCount = cvcamGetCamerasCount();
	if (0 == camCount)
	{
		std::cout << "available camera not found" << std::endl;
		return;
	}
	const int camId = 0;
/*
	int* selectedCamIndexes;
	const int selectedCamCount = cvcamSelectCamera(&selectedCamIndexes);
	if (0 == selectedCamCount)
	{
		std::cout << "any cam failed to be connected" << std::endl;
		return;
	}
	const int camId = selectedCamIndexes[0];
*/
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 176, imageHeight = 144;
	const char *windowName = "capturing from CAM";

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName, imageWidth, imageHeight);

	//
	CvCapture *capture = cvCaptureFromCAM(camId);

	std::cout << "press any key if want to finish ... " << std::endl;
	IplImage *image = NULL;
	while (cvWaitKey(1) < 0)
	{
		//cvGrabFrame(capture);
		//image = cvRetrieveFrame(capture);
		image = cvQueryFrame(capture);

		cvShowImage(windowName, image);
	}
	std::cout << "end capturing ... " << std::endl;

	//
	cvReleaseCapture(&capture);
	cvDestroyWindow(windowName);
}

void capture_image_by_callback()
{
	const int camCount = cvcamGetCamerasCount();
	if (0 == camCount)
	{
		std::cout << "available camera not found" << std::endl;
		return;
	}
	const int camId = 0;
/*
	int* selectedCamIndexes;
	const int selectedCamCount = cvcamSelectCamera(&selectedCamIndexes);
	if (0 == selectedCamCount)
	{
		std::cout << "any cam failed to be connected" << std::endl;
		return;
	}
	const int camId = selectedCamIndexes[0];
*/
	const int imageWidth = 320, imageHeight = 240;
	//const int imageWidth = 176, imageHeight = 144;
	const char* windowName = "cvcam window by callback";

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	HWND hCamWnd = (HWND)cvGetWindowHandle(windowName);

	int retval;

	// camera property
	retval = cvcamSetProperty(camId, CVCAM_PROP_ENABLE, CVCAMTRUE);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_ENABLE, ?) is call" << std::endl;
	retval = cvcamSetProperty(camId, CVCAM_PROP_RENDER, CVCAMTRUE);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_RENDER, ?) is call" << std::endl;
	retval = cvcamSetProperty(camId, CVCAM_PROP_WINDOW, &hCamWnd);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_WINDOW, ?) is call" << std::endl;

	// width & height of window
	retval = cvcamSetProperty(camId, CVCAM_RNDWIDTH, (void*)&imageWidth);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_RNDWIDTH, ?) is call" << std::endl;
	retval = cvcamSetProperty(camId, CVCAM_RNDHEIGHT, (void*)&imageHeight);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_RNDHEIGHT, ?) is call" << std::endl;

	retval = cvcamSetProperty(camId, CVCAM_PROP_CALLBACK, opencv_capture_callback);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_CALLBACK, ?) is call" << std::endl;
/*
	retval = cvcamGetProperty(camId, CVCAM_CAMERAPROPS, NULL);
	if (retval < 0)
		std::cout << "error occurs when cvcamGetProperty(?, CVCAM_CAMERAPROPS, ?) is call" << std::endl;
*/
	//cvSetMouseCallback(windowName, opencvMouseCallback, 0L);

	//
	retval = cvcamInit();
	if (0 == retval)
		std::cout << "error occurs when cvcamInit() is call" << std::endl;
	retval = cvcamStart();
	if (-1 == retval)
		std::cout << "error occurs when cvcamStart() is call" << std::endl;

	cvWaitKey(0);

	retval = cvcamStop();
	assert(0 == retval);
	retval = cvcamExit();
	assert(0 == retval);

	cvDestroyWindow(windowName);
}

void capture_image_by_thread()
{
	const int camCount = cvcamGetCamerasCount();
	if (0 == camCount)
	{
		std::cout << "available camera not found" << std::endl;
		return;
	}
	const int camId = 0;
/*
	int* selectedCamIndexes;
	const int selectedCamCount = cvcamSelectCamera(&selectedCamIndexes);
	if (0 == selectedCamCount)
	{
		std::cout << "any cam failed to be connected" << std::endl;
		return;
	}
	const int camId = selectedCamIndexes[0];
*/
	const int imageWidth = 320, imageHeight = 240;
	//const int imageWidth = 176, imageHeight = 144;

	int retval;

	// camera property
	retval = cvcamSetProperty(camId, CVCAM_PROP_ENABLE, CVCAMTRUE);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_ENABLE, ?) is call" << std::endl;
	retval = cvcamSetProperty(camId, CVCAM_PROP_RENDER, CVCAMTRUE);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_RENDER, ?) is call" << std::endl;
/*
	retval = cvcamSetProperty(camId, CVCAM_PROP_WINDOW, &hCamWnd);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_PROP_WINDOW, ?) is call" << std::endl;
*/
	// width & height of window
	retval = cvcamSetProperty(camId, CVCAM_RNDWIDTH, (void*)&imageWidth);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_RNDWIDTH, ?) is call" << std::endl;
	retval = cvcamSetProperty(camId, CVCAM_RNDHEIGHT, (void*)&imageHeight);
	if (retval < 0)
		std::cout << "error occurs when cvcamSetProperty(?, CVCAM_RNDHEIGHT, ?) is call" << std::endl;
/*
	retval = cvcamGetProperty(camId, CVCAM_CAMERAPROPS, NULL);
	if (retval < 0)
		std::cout << "error occurs when cvcamGetProperty(?, CVCAM_CAMERAPROPS, ?) is call" << std::endl;
*/
	//cvSetMouseCallback(windowName, opencvMouseCallback, 0L);

	//
	retval = cvcamInit();
	if (0 == retval)
		std::cout << "error occurs when cvcamInit() is call" << std::endl;
	retval = cvcamStart();
	if (-1 == retval)
		std::cout << "error occurs when cvcamStart() is call" << std::endl;

	CvSize size;
	size.width = (int)imageWidth;
	size.height = (int)imageHeight;
	isCapturing = true;
    HANDLE hWorkerThread = CreateThread(
		NULL,
		0,
        opencv_capture_thread_proc,
		(void*)&size,
		0,
		NULL
	);
    if (!hWorkerThread)
		std::cout << "capture thread fail to be created" << std::endl;

	cvWaitKey(0);
	while (true) ;

	isCapturing = false;
	while (isThreadTerminated) ;

	CloseHandle(hWorkerThread);

	retval = cvcamStop();
	assert(0 == retval);
	retval = cvcamExit();
	assert(0 == retval);
}

DWORD WINAPI opencv_capture_thread_proc(LPVOID param)
{
	CvSize* imageSize = static_cast<CvSize*>(param);
	if (!imageSize) return 0L;

	const char* windowName = "cvcam window by thread";

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	HWND hCamWnd = (HWND)cvGetWindowHandle(windowName);

	//
	IplImage* frameImg = cvCreateImage(*imageSize, IPL_DEPTH_8U, 3);

	isThreadTerminated = false;
	while (isCapturing)
	{
		//
		//cvcamPause();
		cvcamGetProperty(0, CVCAM_PROP_RAW, &frameImg);
		cvShowImage(windowName, frameImg);
		//cvcamResume();

		Sleep(0);  // it's important
	}

	cvReleaseImage(&frameImg);
	cvDestroyWindow(windowName);

	isThreadTerminated = true;
	return 0;
}

void opencv_capture_callback(IplImage *image)
{
	static int i = 0;
	char filename[256];

	if (i < 1000)
	{
		++i;
		sprintf(filename, "capture%03d.bmp", i % 1000);
		cvSaveImage(filename, image);
	}

	cvWaitKey(1);
}
#elif 0  // OpenCV 2.0 or below
void capture_image_from_cam()
{
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 176, imageHeight = 144;

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

	//
	const char *windowName = "capturing from CAM";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName, imageWidth, imageHeight);

	std::cout << "press any key to exit ... " << std::endl;
	IplImage *frame = NULL;
	IplImage *image = NULL;
	while (capture && cvWaitKey(1) < 0)
	{
		//cvGrabFrame(capture);
		//frame = cvRetrieveFrame(capture);
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

		cvShowImage(windowName, image);
	}
	cvReleaseImage(&image);  image = NULL;
	std::cout << "end capturing ... " << std::endl;

	//
	cvDestroyWindow(windowName);
	cvReleaseCapture(&capture);  capture = NULL;
}

void capture_image_by_callback()
{
	// TODO [check] >> maybe does not support
	throw std::runtime_error("not yet implemented");
}
#else
void capture_image_from_cam()
{
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 176, imageHeight = 144;

	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

/*
	const double &propPosMsec = capture.get(CV_CAP_PROP_POS_MSEC);
	const double &propPosFrames = capture.get(CV_CAP_PROP_POS_FRAMES);
	const double &propPosAviRatio = capture.get(CV_CAP_PROP_POS_AVI_RATIO);
	const double &propFrameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	const double &propFrameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	const double &propFps = capture.get(CV_CAP_PROP_FPS);
	const double &propFourCC = capture.get(CV_CAP_PROP_FOURCC);
	const double &propFrameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
	const double &propFormat = capture.get(CV_CAP_PROP_FORMAT);
	const double &propMode = capture.get(CV_CAP_PROP_MODE);
	const double &propBrightness = capture.get(CV_CAP_PROP_BRIGHTNESS);
	const double &propContrast = capture.get(CV_CAP_PROP_CONTRAST);
	const double &propSaturation = capture.get(CV_CAP_PROP_SATURATION);
	const double &propHue = capture.get(CV_CAP_PROP_HUE);
	const double &propGain = capture.get(CV_CAP_PROP_GAIN);
	const double &propExposure = capture.get(CV_CAP_PROP_EXPOSURE);
	const double &propConvertRGB = capture.get(CV_CAP_PROP_CONVERT_RGB);
	const double &propWhiteBalance = capture.get(CV_CAP_PROP_WHITE_BALANCE);
	const double &propRectification = capture.get(CV_CAP_PROP_RECTIFICATION);
	const double &propMonochrome = capture.get(CV_CAP_PROP_MONOCROME);

	capture.set(CV_CAP_PROP_POS_MSEC, propPosMsec);
	capture.set(CV_CAP_PROP_POS_FRAMES, propPosFrames);
	capture.set(CV_CAP_PROP_POS_AVI_RATIO, propPosAviRatio);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, propFrameWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, propFrameHeight);
	capture.set(CV_CAP_PROP_FPS, propFps);
	capture.set(CV_CAP_PROP_FOURCC, propFourCC);
	capture.set(CV_CAP_PROP_FRAME_COUNT, propFrameCount);
	capture.set(CV_CAP_PROP_FORMAT, propFormat);
	capture.set(CV_CAP_PROP_MODE, propMode);
	capture.set(CV_CAP_PROP_BRIGHTNESS, propBrightness);
	capture.set(CV_CAP_PROP_CONTRAST, propContrast);
	capture.set(CV_CAP_PROP_SATURATION, propSaturation);
	capture.set(CV_CAP_PROP_HUE, propHue);
	capture.set(CV_CAP_PROP_GAIN, propGain);
	capture.set(CV_CAP_PROP_EXPOSURE, propExposure);
	capture.set(CV_CAP_PROP_CONVERT_RGB, propConvertRGB);
	capture.set(CV_CAP_PROP_WHITE_BALANCE, propWhiteBalance);
	capture.set(CV_CAP_PROP_RECTIFICATION, propRectification);
	capture.set(CV_CAP_PROP_MONOCROME, propMonochrome);
*/
	capture.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);

	//
	const std::string windowName("capturing from CAM");
	cv::namedWindow(windowName);
	//cv::resizeWindow(windowName, imageWidth, imageHeight);

	std::cout << "press any key to exit ... " << std::endl;
	cv::Mat frame, image;
	while (cv::waitKey(1) < 0)
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			//break;
			continue;
		}

#if 1
		if (image.empty()) image = frame.clone();
		else frame.copyTo(image);
#else
		image = frame;
#endif
		if (image.empty()) continue;

#if 0
		cv::flip(image, image, 0);  // flip vertically (around x-axis)
#elif 0
		cv::flip(image, image, 1);  // flip horizontally (around y-axis)
#elif 0
		cv::flip(image, image, -1);  // flip vertically & horizontally (around both axes)
#endif

		cv::imshow(windowName, image);
	}
	std::cout << "end capturing ... " << std::endl;

	//
	cv::destroyWindow(windowName);
}

void capture_image_by_callback()
{
	// TODO [check] >> maybe does not support
	throw std::runtime_error("not yet implemented");
}
#endif

void draw_cross(IplImage *img, const int x, const int y, const int len)
{
	const int CONTOUR_COUNT = 1;
	const int POINT_COUNT = 12;
	const int ptCounts[] = { POINT_COUNT };
	CvPoint *contours[CONTOUR_COUNT];

	CvPoint pts1[POINT_COUNT];
	pts1[0].x = x + len;			pts1[0].y = y;
	pts1[1].x = x + 2 * len;		pts1[1].y = y;
	pts1[2].x = x + 2 * len;		pts1[2].y = y + len;
	pts1[3].x = x + 3 * len;		pts1[3].y =  y + len;
	pts1[4].x = x + 3 * len;		pts1[4].y = y + 2 * len;
	pts1[5].x = x + 2 * len;		pts1[5].y = y + 2 * len;
	pts1[6].x = x + 2 * len;		pts1[6].y = y + 3 * len;
	pts1[7].x = x + len;			pts1[7].y = y + 3 * len;
	pts1[8].x = x + len;			pts1[8].y = y + 2 * len;
	pts1[9].x = x;					pts1[9].y = y + 2 * len;
	pts1[10].x = x;					pts1[10].y = y + len;
	pts1[11].x = x + len;			pts1[11].y = y + len;

	contours[0] = pts1;

	cvFillPoly(img, (CvPoint **)contours, ptCounts, 1, CV_RGB(255, 255, 255), 8, 0);
}

void draw_cross(cv::Mat &img, const int x, const int y, const int len)
{
	const int CONTOUR_COUNT = 1;
	const int POINT_COUNT = 12;
	const int ptCounts[] = { POINT_COUNT };
	cv::Point *contours[CONTOUR_COUNT];

	cv::Point pts1[POINT_COUNT];
	pts1[0].x = x + len;			pts1[0].y = y;
	pts1[1].x = x + 2 * len;		pts1[1].y = y;
	pts1[2].x = x + 2 * len;		pts1[2].y = y + len;
	pts1[3].x = x + 3 * len;		pts1[3].y =  y + len;
	pts1[4].x = x + 3 * len;		pts1[4].y = y + 2 * len;
	pts1[5].x = x + 2 * len;		pts1[5].y = y + 2 * len;
	pts1[6].x = x + 2 * len;		pts1[6].y = y + 3 * len;
	pts1[7].x = x + len;			pts1[7].y = y + 3 * len;
	pts1[8].x = x + len;			pts1[8].y = y + 2 * len;
	pts1[9].x = x;					pts1[9].y = y + 2 * len;
	pts1[10].x = x;					pts1[10].y = y + len;
	pts1[11].x = x + len;			pts1[11].y = y + len;

	contours[0] = pts1;

	cv::fillPoly(img, (const cv::Point **)contours, ptCounts, 1, CV_RGB(255, 255, 255), 8, 0, cv::Point());
}

void capture_write_file_from_images()
{
	const std::string VIDEO_FILENAME = "./machine_vision_data/opencv/synthesized_cross_output.avi";

	const int IMAGE_WIDTH = 320, IMAGE_HEIGHT = 240;

#if 0
    const CvSize FRAME_SIZE = cvSize(IMAGE_WIDTH, IMAGE_HEIGHT);
	const double FPS = 30;
	const int isColor = 1;
	CvVideoWriter *videoWriter = cvCreateVideoWriter(
        VIDEO_FILENAME.c_str(),
        CV_FOURCC('X', 'V', 'I', 'D'),
        FPS,
        FRAME_SIZE,
		isColor
	);

	IplImage *img = cvCreateImage(FRAME_SIZE, IPL_DEPTH_8U, 3);

	const int len = 50;
	int x0 = 10, y0 = 10;
	for (int j = 0; j < 5; ++j)
	{
		for (int i = 0; i <= 70; ++i)
		{
			cvSetZero(img);
			draw_cross(cv::Mat(img), x0 + i, y0 + i, len);

			cvWriteFrame(videoWriter, img);
		}

		for (int i = 70; i >= 0; --i)
		{
			cvSetZero(img);
			draw_cross(cv::Mat(img), x0 + i, y0 + i, len);

			cvWriteFrame(videoWriter, img);
		}
	}

	cvReleaseImage(&img);

	if (videoWriter) cvReleaseVideoWriter(&videoWriter);
#else
	const double FPS = 30;
	const cv::Size FRAME_SIZE(IMAGE_WIDTH, IMAGE_HEIGHT);
	const bool isColor = true;
	cv::VideoWriter videoWriter(VIDEO_FILENAME, CV_FOURCC('D', 'I', 'V', 'X'), FPS, FRAME_SIZE, isColor);
	if (!videoWriter.isOpened())
	{
		std::cout << "cv::VideoWriter failed to open" << std::endl;
		return;
	}

	cv::Mat img(FRAME_SIZE, CV_8UC3, cv::Scalar::all(0));

	const int len = 50;
	int x0 = 10, y0 = 10;
	for (int j = 0; j < 5; ++j)
	{
		for (int i = 0; i <= 70; ++i)
		{
			img = cv::Mat::zeros(FRAME_SIZE, CV_8UC3);
			draw_cross(img, x0 + i, y0 + i, len);

			videoWriter << img;
		}

		for (int i = 70; i >= 0; --i)
		{
			img = cv::Mat::zeros(FRAME_SIZE, CV_8UC3);
			draw_cross(img, x0 + i, y0 + i, len);

			videoWriter << img;
		}
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void image_sequence()
{
	//local::capture_image_from_file();
	//local::capture_image_from_cam();

	//local::capture_image_by_callback();

	// TODO [fix] >> it's not working
	//local::capture_image_by_thread();

	local::capture_write_file_from_images();
}

}  // namespace my_opencv
