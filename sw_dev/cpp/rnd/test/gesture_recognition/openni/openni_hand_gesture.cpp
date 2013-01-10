//#include "stdafx.h"
#include <XnCppWrapper.h>
#include <iostream>


namespace {
namespace local {

const XnChar *GESTURE_TO_USE = "Click";

xn::GestureGenerator gestureGen;
xn::HandsGenerator handsGen;
	
void XN_CALLBACK_TYPE gestureRecognizedCallback(xn::GestureGenerator &generator, const XnChar *strGesture, const XnPoint3D *pIDPosition, const XnPoint3D *pEndPosition, void *pCookie)
{
	std::cout << "gesture recognized: " << strGesture << std::endl;
	generator.RemoveGesture(strGesture);
	handsGen.StartTracking(*pEndPosition);
}

void XN_CALLBACK_TYPE gestureProcessCallback(xn::GestureGenerator &generator, const XnChar *strGesture, const XnPoint3D *pPosition, XnFloat fProgress, void *pCookie)
{
}

void XN_CALLBACK_TYPE handCreateCallback(xn::HandsGenerator &generator, XnUserID user, const XnPoint3D *pPosition, XnFloat fTime, void *pCookie)
{
	std::cout << "new hand: " << user << " @ (" << pPosition->X << "," << pPosition->Y << "," << pPosition->Z << ")" << std::endl;
}

void XN_CALLBACK_TYPE handDestroyCallback(xn::HandsGenerator &generator, XnUserID user, XnFloat fTime, void *pCookie)
{
	std::cout << "lost hand: " << user << std::endl;
	gestureGen.AddGesture(GESTURE_TO_USE, NULL);
}

void XN_CALLBACK_TYPE handUpdateCallback(xn::HandsGenerator &generator, XnUserID user, const XnPoint3D *pPosition, XnFloat fTime, void *pCookie)
{
}

int hand_gesture()
{
	XnStatus rc = XN_STATUS_OK;

	// initialize a context object
	xn::Context context;
#if 0
	rc = context.Init();
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to initialize a context object: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
#else
	xn::EnumerationErrors errors;
	rc = context.InitFromXmlFile("./gesture_recognition_data/openni/data/SamplesConfig.xml", &errors);
	if (XN_STATUS_NO_NODE_PRESENT == rc)
	{
		XnChar errStr[1024];
		errors.ToString(errStr, 1024);
		std::cout << "fail to initialize a context object: " << xnGetStatusString(rc) << " ==> " << errStr << std::endl;
		return rc;
	}
	else if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to initialize a context object: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
#endif

	// create a GestureGenerator node
	rc = gestureGen.Create(context);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a GestureGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
	// create a HandGenerator node
	rc = handsGen.Create(context);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a HandGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// register to gesture callbacks
	XnCallbackHandle hGestureCallback;
	rc = gestureGen.RegisterGestureCallbacks(gestureRecognizedCallback, gestureProcessCallback, NULL, hGestureCallback);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to register to gesture callbacks: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
	// register to hands callbacks
	XnCallbackHandle hHandsCallback;
	rc = handsGen.RegisterHandCallbacks(handCreateCallback, handUpdateCallback, handDestroyCallback, NULL, hHandsCallback);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to register to hand callbacks: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// make it start generating data
	rc = context.StartGeneratingAll();
	if (XN_STATUS_OK != rc)
	{
		// ignore errors, "Setting resolution to QVGA"
		// [ref] http://wiki.openni.org/mediawiki/index.php/Main_Page

		std::cout << "fail to start generating data: " << xnGetStatusString(rc) << std::endl;
		//return rc;
	}

	rc = gestureGen.AddGesture(GESTURE_TO_USE, NULL);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to add gesture: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// main loop
	std::cout << "start main loop ..." << std::endl;
	bool shouldRun = true;
	while (shouldRun)
	{
		// wait for new data to be available
		rc = context.WaitAndUpdateAll();
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to update data: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		// TODO [implement] >> process
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

}  // namespace local
}  // unnamed namespace

namespace my_openni {

void hand_gesture()
{
	local::hand_gesture();
}

}  // namespace my_openni
