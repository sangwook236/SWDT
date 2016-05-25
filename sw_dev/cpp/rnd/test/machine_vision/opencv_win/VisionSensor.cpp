#include "stdafx.h"
#include "VisionSensor.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


//----------------------------------------------------------------------------
//

VisionSensor::VisionSensor(const size_t imageWidth, const size_t imageHeight)
: imageWidth_(imageWidth), imageHeight_(imageHeight),
  isInitialized_(false), isCapturing_(false)
{
}

VisionSensor::~VisionSensor()
{
	reset(false);
}

void VisionSensor::reset(bool doesCallInBaseClass /*= true*/)
{
	if (isInitialized_)
		finiSystem();

	isCapturing_ = false;
}
