#include "stdafx.h"
#include "OpenCvVisionSensor.h"

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
//#include <opencv/cvcam.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


//----------------------------------------------------------------------------
//

OpenCvVisionSensor::OpenCvVisionSensor(const size_t imageWidth, const size_t imageHeight)
: base_type(imageWidth, imageHeight), sensorId_(-1), windowHandle_(0L)
#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
  , hWorkerThread_(0L), isWorkerThreadRunning_(false)
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM
{
}

OpenCvVisionSensor::~OpenCvVisionSensor()
{
	reset(false);
}

void OpenCvVisionSensor::reset(bool doesCallInBaseClass /*= true*/)
{
#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
	terminateWorkerThread();
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

	if (doesCallInBaseClass) base_type::reset();
}

void OpenCvVisionSensor::initSystem()
{
	// camera property
	cvcamSetProperty(sensorId_, CVCAM_PROP_ENABLE, CVCAMTRUE);
	cvcamSetProperty(sensorId_, CVCAM_PROP_RENDER, CVCAMTRUE);
	cvcamSetProperty(sensorId_, CVCAM_PROP_WINDOW, windowHandle_);
	
	// width & height of window
	cvcamSetProperty(sensorId_, CVCAM_RNDWIDTH, (void*)&imageWidth_);
	cvcamSetProperty(sensorId_, CVCAM_RNDHEIGHT, (void*)&imageHeight_);

#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
	// callback
	cvcamSetProperty(sensorId_, CVCAM_PROP_CALLBACK, camCallBackForOpenCV);
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

	//cvcamGetProperty(sensorId_, CVCAM_CAMERAPROPS, NULL);

	cvcamInit();

	isInitialized_ = true;
}

void OpenCvVisionSensor::finiSystem()
{
	cvcamExit();

	isInitialized_ = false;
}

void OpenCvVisionSensor::startCapturing()
{
	cvcamStart();

#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
	if (!isWorkerThreadRunning_)
		startWorkerThread();
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

	isCapturing_ = true;
}

void OpenCvVisionSensor::stopCapturing()
{
#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
	terminateWorkerThread();
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

	cvcamStop();

	isCapturing_ = false;
}

#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
/*static*/ void OpenCvVisionSensor::camCallBackForOpenCV(IplImage* image)
{
	cvWaitKey(1);  // it's important
}
#else
/*static*/ DWORD WINAPI OpenCvVisionSensor::captureWorkerThreadProc(LPVOID param)
{
	OpenCvVisionSensor* aVisionSensor = static_cast<OpenCvVisionSensor*>(param);
	if (!aVisionSensor) return 0L;

	const char* windowName = cvGetWindowName(*(HWND*)aVisionSensor->windowHandle_);

	CvSize size;
	size.width = (int)aVisionSensor->getImageWidth();
	size.height = (int)aVisionSensor->getImageHeight();
	IplImage* frameImg = cvCreateImage(size, IPL_DEPTH_8U, 3);

	while (aVisionSensor->isCapturing())
	{
/*
		//cvcamPause();
		cvcamGetProperty(0, CVCAM_PROP_RAW, &frameImg);
		cvShowImage(windowName, frameImg);
		//cvcamResume();
*/
		Sleep(0);  // it's important
	}

	cvReleaseImage(&frameImg);

	aVisionSensor->isWorkerThreadRunning_ = false;
	//aVisionSensor->hWorkerThread_ = 0L;
	return 0;
}

void OpenCvVisionSensor::startWorkerThread()
{
    hWorkerThread_ = CreateThread(
		NULL,
		0, 
        captureWorkerThreadProc,
		(void*)this,
		0,
		NULL
	);  
    if (!hWorkerThread_)
	{
	}

	isWorkerThreadRunning_ = true;
}

void OpenCvVisionSensor::terminateWorkerThread()
{
	int count = 0;
	while (isWorkerThreadRunning_ && count < 1000)
	{
		Sleep(100);
		++count;
	}

	if (isWorkerThreadRunning_)
	{
		TerminateThread(hWorkerThread_, 1);
		isWorkerThreadRunning_ = false;
	}

	hWorkerThread_ = 0L;
}
#endif  // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM
