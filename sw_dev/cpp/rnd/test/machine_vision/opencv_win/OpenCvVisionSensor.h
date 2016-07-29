#if !defined(__MOBILE_TRACER__OPENCV_VISION_SENSOR__H_)
#define __MOBILE_TRACER__OPENCV_VISION_SENSOR__H_ 1


#define __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM 1

#include "VisionSensor.h"

typedef struct _IplImage IplImage;

//----------------------------------------------------------------------------
//

class OpenCvVisionSensor: public VisionSensor
{
public:
	typedef VisionSensor base_type;

public:
	OpenCvVisionSensor(const size_t imageWidth, const size_t imageHeight);
	/*virtual*/ ~OpenCvVisionSensor();

public:
	/*virtual*/ void reset(bool doesCallInBaseClass = true);

	/*virtual*/ void initSystem();
	/*virtual*/ void finiSystem();

	/*virtual*/ void startCapturing();
	/*virtual*/ void stopCapturing();

	void setSensorId(const int sensorId)
	{  sensorId_ = sensorId;  }
	int getSensorId() const
	{  return sensorId_;  }

	void setWindowHandle(void* windowHandle)
	{  windowHandle_ = windowHandle;  }
	void* getWindowHandle()
	{  return windowHandle_;  }

#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
	//bool isWorkerThreadRunning() const  {  return isWorkerThreadRunning_;  }
#endif // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

private:
#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
	static void camCallBackForOpenCV(IplImage*);
#else
	static DWORD WINAPI captureWorkerThreadProc(LPVOID param);

	void startWorkerThread();
	void terminateWorkerThread();
#endif // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM

private:
	int sensorId_;
	void* windowHandle_;

#if __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM == 1
#else
	HANDLE hWorkerThread_;
	bool isWorkerThreadRunning_;
#endif // __USE_CALLBACK_FUNCTION_IN_OPEVCV_CAM
};


#endif  // __MOBILE_TRACER__OPENCV_VISION_SENSOR__H_
