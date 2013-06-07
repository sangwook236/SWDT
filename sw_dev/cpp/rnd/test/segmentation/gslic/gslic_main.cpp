#include "../gslic_lib/FastImgSeg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

// [ref] ${GSLIC_HOME}/testCV/main.cpp
void gslic_sample()
{
	std::list<std::string> rgb_filenames, depth_filenames;
	rgb_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130530T103805.png");
	rgb_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023152.png");
	rgb_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023346.png");
	rgb_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_rgba_20130531T023359.png");
	depth_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130530T103805.png");
	depth_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023152.png");
	depth_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023346.png");
	depth_filenames.push_back("./machine_vision_data/opencv/image_undistortion/kinect_depth_20130531T023359.png");

	//
	FastImgSeg gslic;
	const int num_segments = 1200;
	const SEGMETHOD seg_method = XYZ_SLIC;  // SLIC, RGB_SLIC, XYZ_SLIC
	const double seg_weight = 0.3;

	//
	cv::Mat mask;
	for (std::list<std::string>::iterator it = rgb_filenames.begin(); it != rgb_filenames.end(); ++it)
    {
		cv::Mat img(cv::imread(*it, CV_LOAD_IMAGE_COLOR));
		if (img.empty())
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		// gSLIC currently only support 4-dimensional image
		unsigned char *imgBuffer = new unsigned char [img.cols * img.rows * 4];
		memset(imgBuffer, 0, sizeof(unsigned char) * img.cols * img.rows * 4);

		unsigned char *ptr = imgBuffer;
		for (int i = 0; i < img.rows; ++i)
			for (int j = 0; j < img.cols; ++j)
			{
				const cv::Vec3b &bgr = img.at<cv::Vec3b>(i, j);

				*ptr++ = bgr[0];
				*ptr++ = bgr[1];
				*ptr++ = bgr[2];
				++ptr;
			}

		//
		gslic.initializeFastSeg(img.cols, img.rows, num_segments);

		gslic.LoadImg(imgBuffer);
		gslic.DoSegmentation(seg_method, seg_weight);
		gslic.Tool_GetMarkedImg();  // required for display

		delete [] imgBuffer;
		imgBuffer = NULL;

		//
		ptr = gslic.markedImg;
		for (int i = 0; i < img.rows; ++i)
			for (int j = 0; j < img.cols; ++j)
			{
#if 0
				const unsigned char b = *ptr++;
				const unsigned char g = *ptr++;
				const unsigned char r = *ptr++;
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
				++ptr;
#else
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(*ptr, *(ptr + 1), *(ptr + 2));
				ptr += 4;
#endif
			}

		cv::imshow("segmentation by gSLIC", img);

#if 0
		// segment indexes are stored in FastImgSeg::segMask
		cv::Mat mask_tmp(img.size(), CV_32SC1, (void *)gslic.segMask);
        double minVal = 0.0, maxVal = 0.0;
		cv::minMaxLoc(mask_tmp, &minVal, &maxVal);
		mask_tmp.convertTo(mask, CV_32FC1, 1.0 / maxVal, 0.0);

		cv::imshow("segmentation by gSLIC - mas", mask);
#endif

		const unsigned char key = cv::waitKey(0);
		if (27 == key)
			break;
	}

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_gslic {

}  // namespace my_gslic

int gslic_main (int argc, char *argv[])
{
	bool canUseGPU = false;
	try
	{
		if (cv::gpu::getCudaEnabledDeviceCount() > 0)
		{
			canUseGPU = true;
			std::cout << "GPU info:" << std::endl;
			cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
		}
		else
			std::cout << "GPU not found ..." << std::endl;

		local::gslic_sample();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
