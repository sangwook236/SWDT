//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void openni_interface()
{
	cv::VideoCapture capture(CV_CAP_OPENNI);
	if (!capture.isOpened())
	{
		std::cout << "an OpenNI sensor not found" << std::endl;
		return;
	}

    // CV_CAP_OPENNI_VGA_30HZ, CV_CAP_OPENNI_SXGA_15HZ, CV_CAP_OPENNI_SXGA_30HZ
    // The following modes are only supported by the Xtion Pro Live
    // CV_CAP_OPENNI_QVGA_30HZ, CV_CAP_OPENNI_QVGA_60HZ
	if (!capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ))
		std::cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << std::endl;

	// Print some avalible device settings.
	std::cout << "\nDepth generator output mode:" << std::endl <<
		"\tFRAME_WIDTH      " << capture.get(CV_CAP_PROP_FRAME_WIDTH) << std::endl <<
		"\tFRAME_HEIGHT     " << capture.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl <<
		"\tFRAME_MAX_DEPTH  " << capture.get(CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH) << " mm" << std::endl <<
		"\tFPS              " << capture.get(CV_CAP_PROP_FPS) << std::endl <<
		"\tREGISTRATION     " << capture.get(CV_CAP_PROP_OPENNI_REGISTRATION) << std::endl;

	if (capture.get(CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT))
	{
		std::cout << "\nImage generator output mode:" << std::endl <<
			"\tFRAME_WIDTH   " << capture.get(CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_FRAME_WIDTH) << std::endl <<
			"\tFRAME_HEIGHT  " << capture.get(CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_FRAME_HEIGHT) << std::endl <<
			"\tFPS           " << capture.get(CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_FPS) << std::endl;
	}
	else
	{
		std::cout << "\nDevice doesn't contain image generator." << std::endl;
	}

	//
	std::cout << "press any key if want to finish ..." << std::endl;
	cv::Mat frameRGB, frameDepth, framePointCloud, frameDisparity, frameValidDepthMask;
	while (true)
	{
		if (!capture.grab())
        {
            std::cout << "can not grab images." << std::endl;
            break;
        }

		// CV_CAP_OPENNI_BGR_IMAGE, CV_CAP_OPENNI_GRAY_IMAGE
		capture.retrieve(frameRGB, CV_CAP_OPENNI_BGR_IMAGE);
		if (frameRGB.empty())
		{
			std::cout << "an RGB frame not found ..." << std::endl;
			break;
			//continue;
		}
		capture.retrieve(frameDepth, CV_CAP_OPENNI_DEPTH_MAP);
		if (frameDepth.empty())
		{
			std::cout << "a depth frame not found ..." << std::endl;
			break;
			//continue;
		}
		capture.retrieve(framePointCloud, CV_CAP_OPENNI_POINT_CLOUD_MAP);
		if (frameDepth.empty())
		{
			std::cout << "a depth frame not found ..." << std::endl;
			break;
			//continue;
		}
		// CV_CAP_OPENNI_DISPARITY_MAP, CV_CAP_OPENNI_DISPARITY_MAP_32F
		capture.retrieve(frameDisparity, CV_CAP_OPENNI_DISPARITY_MAP);
		if (frameDepth.empty())
		{
			std::cout << "a depth frame not found ..." << std::endl;
			break;
			//continue;
		}
		capture.retrieve(frameValidDepthMask, CV_CAP_OPENNI_VALID_DEPTH_MASK);
		if (frameDepth.empty())
		{
			std::cout << "a depth frame not found ..." << std::endl;
			break;
			//continue;
		}

		cv::imshow("OpenNI RGB image", frameRGB);
		cv::imshow("OpenNI depth map", frameDepth);
		cv::imshow("OpenNI point cloud map", framePointCloud);
		cv::imshow("OpenNI disparity map", frameDisparity);
		cv::imshow("OpenNI valid depth mask", frameValidDepthMask);

		if (cv::waitKey(1) >= 0)
			break;
	}
	std::cout << "end capturing ..." << std::endl;

	cv::destroyAllWindows();
}

}  // namespace my_opencv
