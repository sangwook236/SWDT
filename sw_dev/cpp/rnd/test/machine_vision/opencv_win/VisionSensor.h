#if !defined(__MOBILE_TRACER__VISION_SENSOR__H_)
#define __MOBILE_TRACER__VISION_SENSOR__H_ 1


//----------------------------------------------------------------------------
//

class VisionSensor
{
protected:
	VisionSensor(const size_t imageWidth, const size_t imageHeight);
public:
	virtual ~VisionSensor();

public:
	virtual void reset(bool doesCallInBaseClass = true);

	virtual void initSystem() = 0;
	virtual void finiSystem() = 0;
	bool isInitialized() const  {  return isInitialized_;  }

	virtual void startCapturing() = 0;
	virtual void stopCapturing() = 0;

	//void setCapturing(bool isCapturing)  {  isCapturing_ = isCapturing;  }
	bool isCapturing() const  {  return isCapturing_;  }

	size_t getImageWidth() const  {  return imageWidth_;  }
	size_t getImageHeight() const  {  return imageHeight_;  }

protected:
	const size_t imageWidth_, imageHeight_;

	bool isInitialized_;
	bool isCapturing_;
};


#endif  // __MOBILE_TRACER__VISION_SENSOR__H_
