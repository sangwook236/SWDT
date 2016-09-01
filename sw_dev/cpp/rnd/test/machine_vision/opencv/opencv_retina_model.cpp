//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/bioinspired.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

// ${OPENCV_HOME}/samples/cpp/retinaDemo.cpp
// ${OPENCV_HOME}/sample/cpp/tutorial_code/contrib/retina_tutorial.cpp
void retina_model()
{
    const bool useLogSampling = true;

	cv::Mat inputFrame;
	cv::VideoCapture videoCapture;

#if 1
	// still image processing.

	const std::string img_filename("./data/machine_vision/opencv/lena_rgb.bmp");

	inputFrame = cv::imread(img_filename, 1);  // load image in RGB mode.
	if (inputFrame.empty())
	{
		std::cerr << "an image file not found: " << img_filename << std::endl;
		return;
	}
#elif 0
    // video processing.

	const std::string video_filename("./data/machine_vision/opencv/tree.avi");

	videoCapture.open(video_filename);
	if (!videoCapture.isOpened())
	{
		std::cerr << "a video file not found: " << video_filename << std::endl;
		return;
	}

	videoCapture >> inputFrame;
#elif 0
    // live video processing.

	const int camId = -1;
	videoCapture.open(camId);
	if (!videoCapture.isOpened())
	{
		std::cerr << "a vision sensor not found: " << camId << std::endl;
		return;
	}

	videoCapture >> inputFrame;
#endif

    if (inputFrame.empty())
    {
        std::cerr << "Input media could not be loaded, aborting" << std::endl;
        return;
    }

	// create a retina instance with default parameters setup, uncomment the initialisation you wanna test
	cv::Ptr<cv::bioinspired::Retina> retina;

	// if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
	if (useLogSampling)
	{
		const bool colorMode = true;
		const int colorSamplingMethod = cv::bioinspired::RETINA_COLOR_BAYER;
		const double reductionFactor = 2.0;
		const double samplingStrenght = 10.0;

		retina = cv::bioinspired::createRetina(inputFrame.size(), colorMode, colorSamplingMethod, useLogSampling, 2.0, samplingStrenght);
	}
	else  // allocate "classical" retina.
		retina = cv::bioinspired::createRetina(inputFrame.size());

#if 0
	// save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup".
	retina->write("./data/machine_vision/opencv/RetinaDefaultParameters.xml");
#endif

#if 0
	// load parameters if file exists.
	retina->setup("./data/machine_vision/opencv/RetinaSpecificParameters.xml");
	retina->clearBuffers();
#endif

	// declare retina output buffers.
	cv::Mat retinaOutput_parvo;
	cv::Mat retinaOutput_magno;

	// processing loop with stop condition.
	while (true)
	{
		// if using video stream, then, grabbing a new frame, else, input remains the same.
		if (videoCapture.isOpened())
			videoCapture >> inputFrame;

		// run retina filter.
		retina->run(inputFrame);

		// retrieve and display retina output.
		retina->getParvo(retinaOutput_parvo);
		retina->getMagno(retinaOutput_magno);

		cv::imshow("retina model - input", inputFrame);
		cv::imshow("retina model - Parvo", retinaOutput_parvo);
		cv::imshow("retina model - Magno", retinaOutput_magno);

		if (cv::waitKey(1) >= 0)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace my_opencv
