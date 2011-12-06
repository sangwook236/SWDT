#if !defined(__MOBILE_TRACER__BUMBLEBEE_VISION_SENSOR__H_)
#define __MOBILE_TRACER__BUMBLEBEE_VISION_SENSOR__H_ 1


#pragma comment(lib, "triclops.lib")
#pragma comment(lib, "digiclops.lib")
#pragma comment(lib, "pgrcameragui.lib")

#include "VisionSensor.h"

#include "triclops.h"
#include "digiclops.h"
#include "pnmutils.h"


//----------------------------------------------------------------------------
//

class BumbleBeeVisionSensor: public VisionSensor
{
public:
	typedef VisionSensor base_type;

public:
	BumbleBeeVisionSensor(const size_t imageWidth, const size_t imageHeight);
	/*virtual*/ ~BumbleBeeVisionSensor();

public:
	/*virtual*/ void reset();

	/*virtual*/ void initSystem();
	/*virtual*/ void finiSystem();

	/*virtual*/ void startCapturing();
	/*virtual*/ void stopCapturing();

	//bool isWorkerThreadRunning() const  {  return isWorkerThreadRunning_;  }

private:
	static DWORD WINAPI captureWorkerThreadProc(LPVOID param);

	void startWorkerThread();
	void terminateWorkerThread();

private:
	//TriclopsInput stereoData_;

	TriclopsInput colorData_;
	//TriclopsImage16 depthImage16_;
	TriclopsColorImage colorImage_;
	TriclopsContext triclops_;
	DigiclopsContext digiclops_;

	HANDLE hWorkerThread_;
	bool isWorkerThreadRunning_;
};


#endif  // __MOBILE_TRACER__BUMBLEBEE_VISION_SENSOR__H_
