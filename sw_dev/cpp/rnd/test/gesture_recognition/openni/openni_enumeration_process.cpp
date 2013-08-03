//#include "stdafx.h"
#include <XnCppWrapper.h>
#include <iostream>


namespace {
namespace local {

int enumeration_process()
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

	// build a query object
	xn::Query query;
	rc = query.SetVendor("PrimeSense");
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to set the requested vendor: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	query.AddSupportedCapability(XN_CAPABILITY_SKELETON);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to add a skeleton capability: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// enumerate
	xn::NodeInfoList possibleChains;
	rc = context.EnumerateProductionTrees(XN_NODE_TYPE_USER, &query, possibleChains, NULL);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to enumerate production trees: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	if (possibleChains.IsEmpty())
	{
		std::cout << "fail to search for any production tree" << std::endl;
		return 1;
	}

	// no errors so far. this means list has at least one item. take the first one.
	xn::NodeInfo selected = *possibleChains.Begin();
	XnProductionNodeDescription desc = selected.GetDescription();
	std::cout << "name: " << desc.strName << ", type: " << desc.Type << ", vendor: " << desc.strVendor << std::endl;

	// create it
	rc = context.CreateProductionTree(selected);  // Oops !!! runtime error
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a selected production tree: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// take the node
	xn::UserGenerator userGen;
	rc = selected.GetInstance(userGen);
	if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to get a user generator node: " << xnGetStatusString(rc) << std::endl;
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
		rc = context.WaitOneUpdateAll(userGen);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to update data: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		//
		XnUserID users[10];
		XnUInt16 userNum = 10;
		rc = userGen.GetUsers(users, userNum);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to get the current users: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		xn::SceneMetaData sceneMD;
		rc = userGen.GetUserPixels(users[0], sceneMD);
		if (XN_STATUS_OK != rc)
		{
			std::cout << "fail to get the pixels that belong to a user: " << xnGetStatusString(rc) << std::endl;
			continue;
		}

		// TODO [implement] >> process user pixels
	}
	std::cout << "end main loop ..." << std::endl;

	// clean-up
	context.Shutdown();

	return 0;
}

int enumeration_error()
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

	// create a HandGenerator node
	//xn::EnumerationErrors errors;
	xn::HandsGenerator handsGen;
	rc = context.CreateAnyProductionTree(XN_NODE_TYPE_HANDS, NULL, handsGen, &errors);
	if (XN_STATUS_NO_NODE_PRESENT == rc)
	{
		// iterate over enumeration errors, and print each one
		for (xn::EnumerationErrors::Iterator it = errors.Begin(); it != errors.End(); ++it)
		{
			XnChar description[512];
			xnProductionNodeDescriptionToString(&it.Description(), description, 512);
			std::cout << description << " failed to enumerate: " << xnGetStatusString(it.Error()) << std::endl;
		}
		return rc;
	}
	else if (XN_STATUS_OK != rc)
	{
		std::cout << "fail to create a HandGenerator node: " << xnGetStatusString(rc) << std::endl;
		return rc;
	}

	// clean-up
	context.Shutdown();

	return 0;
}

}  // namespace local
}  // unnamed namespace

namespace my_openni {

void enumeration_process()
{
	//local::enumeration_process();
	local::enumeration_error();
}

}  // namespace my_openni
