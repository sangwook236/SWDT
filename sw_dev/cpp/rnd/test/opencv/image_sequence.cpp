#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include <opencv/cvcam.h>
#include <iostream>
#include <cassert>
#include <cstdlib>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//#define __USE_OPENCV_1_0 1

namespace {

bool isCapturing = true, isThreadTerminated = false;

void capture_image_from_file()
{
	const int imageWidth = 640, imageHeight = 480;
	//const int imageWidth = 176, imageHeight = 144;
	const char *windowName = "capturing from file";

	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(windowName, imageWidth, imageHeight);

	//
	const std::string avi_filename("opencv_data\\flycap-0001.avi");
	//CvCapture *capture = cvCaptureFromFile(avi_filename.c_str());
	CvCapture *capture = cvCreateFileCapture(avi_filename.c_str());

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

#if __USE_OPENCV_1_0
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
#else
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

	cvFillPoly(img, (CvPoint**)contours, ptCounts, 1, CV_RGB(255, 255, 255), 8, 0);
}

void capture_write_file_from_images()
{
	const std::string avi_filename = "opencv_data\\synthesized_cross.avi";

	const int imgWidth = 320, imgHeight = 240;
    CvSize imgSize = cvSize(imgWidth, imgHeight);

	CvVideoWriter *avi_writer = cvCreateVideoWriter(
        avi_filename.c_str(),
        CV_FOURCC('x', 'v', 'i', 'd'),
        25,
        imgSize
	);

	const int len = 50;
	int x0 = 10, y0 = 10;
	for (int j = 0; j < 5; ++j)
	{
		for (int i = 0; i <= 70; ++i)
		{
			IplImage *img = cvCreateImage(imgSize, IPL_DEPTH_8U, 3);
			cvSetZero(img);
			draw_cross(img, x0 + i, y0 + i, len);

			cvWriteFrame(avi_writer, img);

			cvReleaseImage(&img);
		}

		for (int i = 70; i >= 0; --i)
		{
			IplImage *img = cvCreateImage(imgSize, IPL_DEPTH_8U, 3);
			cvSetZero(img);
			draw_cross(img, x0 + i, y0 + i, len);

			cvWriteFrame(avi_writer, img);

			cvReleaseImage(&img);
		}
	}

	if (avi_writer) cvReleaseVideoWriter(&avi_writer);
}

}  // unnamed namespace

void image_sequence()
{
	//capture_image_from_file();
	capture_image_from_cam();

	//capture_image_by_callback();

	// TODO [] : it's not working
	//capture_image_by_thread();

	//capture_write_file_from_images();
}
