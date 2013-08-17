//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vigra {

void seeded_region_growing();
void watershed_region_growing();
void watershed();
void slic();

}  // namespace my_vigra

int vigra_main(int argc, char *argv[])
{
	try
	{
		cv::theRNG();

#if 0
		if (cv::gpu::getCudaEnabledDeviceCount() > 0)
		{
			std::cout << "GPU info:" << std::endl;
			cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
		}
		else
			std::cout << "GPU not found ..." << std::endl;
#endif

		// segmentation ---------------------------------------------
		{
			// seeded region growing (SRG).
			my_vigra::seeded_region_growing();

			// watershed region growing.
			//my_vigra::watershed_region_growing();

			// watershed.
			//my_vigra::watershed();

			// simple linear iterative clustering (SLIC) - superpixel.
			//my_vigra::slic();
		}
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
