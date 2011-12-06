#include "stdafx.h"
#include <XnCppWrapper.h>
#include <iostream>


namespace {
namespace local {

const XnChar *POSE_TO_USE = "Psi";

xn::UserGenerator userGen;
	
void XN_CALLBACK_TYPE userNewUserCallback(xn::UserGenerator &generator, XnUserID user, void *pCookie)
{
	std::cout << "new user: " << user << std::endl;
	generator.GetPoseDetectionCap().StartPoseDetection(POSE_TO_USE, user);
}
	
void XN_CALLBACK_TYPE userLostUserCallback(xn::UserGenerator &generator, XnUserID user, void *pCookie)
{
}

void XN_CALLBACK_TYPE poseStartCallback(xn::PoseDetectionCapability &capability, const XnChar *strPose, XnUserID user, void *pCookie)
{
	std::cout << "pose " << strPose << " for user " << user << std::endl;
	userGen.GetPoseDetectionCap().StopPoseDetection(user);
	userGen.GetSkeletonCap().RequestCalibration(user, TRUE);
}

void XN_CALLBACK_TYPE poseEndCallback(xn::PoseDetectionCapability &capability, const XnChar *strPose, XnUserID user, void *pCookie)
{
}
	
void XN_CALLBACK_TYPE calibrationStartCallback(xn::SkeletonCapability &capability, XnUserID user, void *pCookie)
{
	std::cout << "starting calibration for user " << user << std::endl;
}
	
void XN_CALLBACK_TYPE calibrationEndCallback(xn::SkeletonCapability &capability, XnUserID user, XnBool bSuccess, void *pCookie)
{
	if (bSuccess)
	{
		std::cout << "user calibrated" << std::endl;
		userGen.GetSkeletonCap().StartTracking(user);
	}
	else
	{
		std::cout << "failed to calibrate user " << user << std::endl;
		userGen.GetPoseDetectionCap().StartPoseDetection(POSE_TO_USE, user);
	}
}

int skeleton()
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
	rc = context.InitFromXmlFile("./openni_data/data/SamplesConfig.xml", &errors);
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

	// create a UserGenerator node
	rc = userGen.Create(context);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a UserGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// register to user callbacks
	XnCallbackHandle hUserCallback;
	rc = userGen.RegisterUserCallbacks(userNewUserCallback, userLostUserCallback, NULL, hUserCallback);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to register to gesture callbacks: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
	// register to pose callbacks
	XnCallbackHandle hPoseCallback;
	rc = userGen.GetPoseDetectionCap().RegisterToPoseCallbacks(poseStartCallback, poseEndCallback, NULL, hPoseCallback);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to register to pose callbacks: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}
	// register to calibration callbacks
	XnCallbackHandle hCalibrationCallback;
	rc = userGen.GetSkeletonCap().RegisterCalibrationCallbacks(calibrationStartCallback, calibrationEndCallback, NULL, hCalibrationCallback);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to register to pose callbacks: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// set the profile
	rc = userGen.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to set the profile: " << xnGetStatusString(rc) << std::endl;
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

		// extract head position of each tracked user
		XnUserID users[15];
		XnUInt16 userNum = 15;
		userGen.GetUsers(users, userNum);
		for (int i = 0; i < userNum; ++i)
		{
			if (userGen.GetSkeletonCap().IsTracking(users[i]))
			{
				XnSkeletonJointPosition head;
				userGen.GetSkeletonCap().GetSkeletonJointPosition(users[i], XN_SKEL_HEAD, head);
				std::cout << users[i] << "(" << head.position.X << "," << head.position.Y << "," << head.position.Z << ") [" << head.fConfidence << "]" << std::endl;
			}
		}

		// TODO [implement] >> process
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

}  // local
}  // unnamed namespace

void skeleton()
{
	local::skeleton();
}
