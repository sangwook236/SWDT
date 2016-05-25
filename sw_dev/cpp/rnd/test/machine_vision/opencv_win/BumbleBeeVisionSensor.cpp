#include "stdafx.h"
#include "BumbleBeeVisionSensor.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


//----------------------------------------------------------------------------
//

BumbleBeeVisionSensor::BumbleBeeVisionSensor(const size_t imageWidth, const size_t imageHeight)
: base_type(imageWidth, imageHeight), hWorkerThread_(0L), isWorkerThreadRunning_(false)
{
}

BumbleBeeVisionSensor::~BumbleBeeVisionSensor()
{
}

void BumbleBeeVisionSensor::reset()
{
	base_type::reset();

	terminateWorkerThread();
}

void BumbleBeeVisionSensor::initSystem()
{
	// open the Digiclops
	digiclopsCreateContext(&digiclops_);
	digiclopsInitialize(digiclops_, 0);
	
	// get the camera module configuration
	digiclopsGetTriclopsContextFromCamera(digiclops_, &triclops_);

	// set the digiclops to deliver the stereo image and right (color) image
	digiclopsSetImageTypes(digiclops_, STEREO_IMAGE | RIGHT_IMAGE);

	// set the Digiclops resolution
	// use 'HALF' resolution when you need faster throughput, especially for
	// color images
	// digiclopsSetImageResolution(digiclops_, DIGICLOPS_HALF);
	digiclopsSetImageResolution(digiclops_, DIGICLOPS_FULL);
	
	// preprocessing the images
	//triclopsPreprocess(triclops_, &stereoData_) ;
	
	// stereo processing
	//triclopsStereo(triclops_) ;

	isInitialized_ = true;
}

void BumbleBeeVisionSensor::finiSystem()
{
	// destroy the digiclops context
	digiclopsDestroyContext(digiclops_);
	// destroy the triclops context
	triclopsDestroyContext(triclops_);

	isInitialized_ = false;
}

void BumbleBeeVisionSensor::startCapturing()
{
	// get the camera module configuration
	digiclopsGetTriclopsContextFromCamera(digiclops_, &triclops_);
	
	// set the digiclops to deliver the stereo image and right (color) image
	digiclopsSetImageTypes(digiclops_, STEREO_IMAGE | RIGHT_IMAGE);
	
	// set the Digiclops resolution
	// use 'HALF' resolution when you need faster throughput, especially for
	// color images
	// digiclopsSetImageResolution(digiclops_, DIGICLOPS_HALF);
	digiclopsSetImageResolution(digiclops_, DIGICLOPS_FULL);
	
	// preprocessing the images
	//triclopsPreprocess(triclops_, &stereoData_) ;
	
	// stereo processing
	//triclopsStereo(triclops_) ;

	// start grabbing
	digiclopsStart(digiclops_);
	
	// set up some stereo parameters:
	// set to 320x240 output images
	triclopsSetResolution(triclops_, imageWidth_, imageHeight_);
/*   
	// set disparity range
	triclopsSetDisparity(triclops_, 1, 100);

	triclopsSetStereoMask(triclops_, 11);
	triclopsSetEdgeCorrelation(triclops_, 1);
	triclopsSetEdgeMask(triclops_, 11);

	// lets turn off all validation except subpixel and surface
	// this works quite well
	triclopsSetTextureValidation(triclops_, 0);
	triclopsSetUniquenessValidation(triclops_, 0);

	// turn on sub-pixel interpolation
	triclopsSetSubpixelInterpolation(triclops_, 1) ;
	// make sure strict subpixel validation is on
	triclopsSetStrictSubpixelValidation(triclops_, 1);

	// turn on surface validation
	triclopsSetSurfaceValidation(triclops_, 1);
	triclopsSetSurfaceValidationSize(triclops_, 200);
	triclopsSetSurfaceValidationDifference(triclops_, 0.5);
*/

	if (!isWorkerThreadRunning_)
		startWorkerThread();

	isCapturing_ = true;
}

void BumbleBeeVisionSensor::stopCapturing()
{
	terminateWorkerThread();

	// close the digiclops
	digiclopsStop(digiclops_);

	isCapturing_ = false;
}

/*static*/ DWORD WINAPI BumbleBeeVisionSensor::captureWorkerThreadProc(LPVOID param)
{
	BumbleBeeVisionSensor* aVisionSensor = static_cast<BumbleBeeVisionSensor*>(param);
	if (!aVisionSensor) return 0L;

	const char windowName[] = "Experiment #4";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(windowName, mouseCallbackForOpenCV, (void*)aVisionSensor);

	CvSize size;
	size.width = aVisionSensor->getImageWidth();
	size.height = aVisionSensor->getImageHeight();
	IplImage* img = cvCreateImage(size, IPL_DEPTH_8U, 3);

	size_t oldPtX = -1, oldPtY = -1;
	while (aVisionSensor->isCapturing())
	{
		cvWaitKey(1);
	}

	cvReleaseImage(&img);
	cvDestroyWindow(windowName);

	aVisionSensor->isWorkerThreadRunning_ = false;
	//aVisionSensor->hWorkerThread_ = 0L;
	return 0;
}

void BumbleBeeVisionSensor::startWorkerThread()
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

void BumbleBeeVisionSensor::terminateWorkerThread()
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
