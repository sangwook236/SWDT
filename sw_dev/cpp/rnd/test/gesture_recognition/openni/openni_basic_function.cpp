//#include "stdafx.h"
#include <XnCppWrapper.h>
#include <iostream>


namespace {
namespace local {

int basic_function()
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
	rc = context.InitFromXmlFile("./data/gesture_recognition/openni/data/SamplesConfig.xml", &errors);
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

	// create a DepthGenerator node
#if 0
	xn::DepthGenerator depthGen;
	rc = depthGen.Create(context);
#else
	xn::DepthGenerator depthGen;
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGen);
#endif
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a DepthGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// create an ImageGenerator node
#if 0
	xn::ImageGenerator imageGen;
	rc = imageGen.Create(context);
#else
	xn::ImageGenerator imageGen;
	rc = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGen);
#endif
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create an ImageGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	//
	{
		xn::DepthMetaData depthMD;
		depthGen.GetMetaData(depthMD);
		xn::ImageMetaData imageMD;
		imageGen.GetMetaData(imageMD);

		if (imageMD.FullXRes() != depthMD.FullXRes() || imageMD.FullYRes() != depthMD.FullYRes())
		{
			std::cout << "the device depth and image resolution must be equal" << std::endl;
			return 1;
		}

		if (imageMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
		{
			std::cout << "the device image format must be RGB24" << std::endl;
			return 1;
		}
	}

	//
	{
		XnMapOutputMode mapMode;
		mapMode.nXRes = XN_VGA_X_RES;
		mapMode.nYRes = XN_VGA_Y_RES;
		mapMode.nFPS = 30;
		rc = depthGen.SetMapOutputMode(mapMode);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to set map output mode: " << xnGetStatusString(rc) << std::endl;
			return rc;
		}
	}

	// make it start generating data
	rc = context.StartGeneratingAll();
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to start generating data: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// main loop
	std::cout << "start main loop ..." << std::endl;
	bool shouldRun = true;
	while (shouldRun)
	{
		// wait for new data to be available
		rc = context.WaitOneUpdateAll(depthGen);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to update data: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		// take current depth map
		const XnDepthPixel *depthMap = depthGen.GetDepthMap();
		// take current image map
		const XnUInt8 *imageMap = imageGen.GetImageMap();

		// TODO [implement] >> process depth & image maps
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

int record_data()
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
	rc = context.InitFromXmlFile("./data/gesture_recognition/openni/data/SamplesConfig.xml", &errors);
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

	// create a DepthGenerator node
	xn::DepthGenerator depthGen;
	rc = depthGen.Create(context);

	// make it start generating data
	rc = context.StartGeneratingAll();
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to start generating data: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// create recoder
	xn::Recorder recorder;
	rc = recorder.Create(context);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a recorder: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// initialize it
	rc = recorder.SetDestination(XN_RECORD_MEDIUM_FILE, "./data/gesture_recognition/openni/tmp_recorder.oni");
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to set destination: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// add depth node to recording
	rc = recorder.AddNodeToRecording(depthGen, XN_CODEC_16Z_EMB_TABLES);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to add a node to recording: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// main loop
	std::cout << "start main loop ..." << std::endl;
	bool shouldRun = true;
	while (shouldRun)
	{
		// wait for new data to be available
		rc = context.WaitOneUpdateAll(depthGen);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to update data: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		// take current depth map
		const XnDepthPixel *depthMap = depthGen.GetDepthMap();

		// TODO [implement] >> process depth map
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

int play_data()
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
	rc = context.InitFromXmlFile("./data/gesture_recognition/openni/data/SamplesConfig.xml", &errors);
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

	// open recording
	rc = context.OpenFileRecording("./data/gesture_recognition/openni/tmp_recorder.oni");
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to open recording: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// create a DepthGenerator node
	xn::DepthGenerator depthGen;
	rc = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGen);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a DepthGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// make it start generating data
	rc = context.StartGeneratingAll();
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to start generating data: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// main loop
	std::cout << "start main loop ..." << std::endl;
	bool shouldRun = true;
	while (shouldRun)
	{
		// wait for new data to be available
		rc = context.WaitOneUpdateAll(depthGen);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to update data: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		// take current depth map
		const XnDepthPixel *depthMap = depthGen.GetDepthMap();

		// TODO [implement] >> process depth map
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

}  // namespace local
}  // unnamed namespace

namespace my_openni {

void basic_function()
{
	//local::basic_function();

	local::record_data();
	//local::play_data();
}

}  // namespace my_openni
