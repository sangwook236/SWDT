#include "../bgslibrary_lib/package_bgs/FrameDifferenceBGS.h"
#include "../bgslibrary_lib/package_bgs/StaticFrameDifferenceBGS.h"
#include "../bgslibrary_lib/package_bgs/WeightedMovingMeanBGS.h"
#include "../bgslibrary_lib/package_bgs/WeightedMovingVarianceBGS.h"
#include "../bgslibrary_lib/package_bgs/MixtureOfGaussianV1BGS.h"
#include "../bgslibrary_lib/package_bgs/MixtureOfGaussianV2BGS.h"
#include "../bgslibrary_lib/package_bgs/AdaptiveBackgroundLearning.h"
#include "../bgslibrary_lib/package_bgs/GMG.h"

#include "../bgslibrary_lib/package_bgs/dp/DPAdaptiveMedianBGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPGrimsonGMMBGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPZivkovicAGMMBGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPMeanBGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPWrenGABGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPPratiMediodBGS.h"
#include "../bgslibrary_lib/package_bgs/dp/DPEigenbackgroundBGS.h"

#include "../bgslibrary_lib/package_bgs/tb/T2FGMM_UM.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FGMM_UV.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FMRF_UM.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FMRF_UV.h"
#include "../bgslibrary_lib/package_bgs/tb/FuzzySugenoIntegral.h"
#include "../bgslibrary_lib/package_bgs/tb/FuzzyChoquetIntegral.h"

#include "../bgslibrary_lib/package_bgs/jmo/MultiLayerBGS.h"

#include "../bgslibrary_lib/package_bgs/pt/PixelBasedAdaptiveSegmenter.h"

#include "../bgslibrary_lib/package_bgs/lb/LBSimpleGaussian.h"
#include "../bgslibrary_lib/package_bgs/lb/LBFuzzyGaussian.h"
#include "../bgslibrary_lib/package_bgs/lb/LBMixtureOfGaussians.h"
#include "../bgslibrary_lib/package_bgs/lb/LBAdaptiveSOM.h"
#include "../bgslibrary_lib/package_bgs/lb/LBFuzzyAdaptiveSOM.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cassert>


namespace {
namespace local {

void demo()
{
#if 0
	const std::string avi_filename("./change_detection_data/video.avi");
	cv::VideoCapture capture(avi_filename);
#else
	const int camId = -1;
	cv::VideoCapture capture(camId);
#endif
	if (!capture.isOpened())
	{
		std::cout << "a vision sensor not found" << std::endl;
		return;
	}

	const int whichAlgorithm = 16;
	bool useGrayImage = true;
	IBGS *bgs = NULL;

	// background subtraction methods
	if (1 == whichAlgorithm) bgs = new FrameDifferenceBGS;
	if (2 == whichAlgorithm) bgs = new StaticFrameDifferenceBGS;
	if (3 == whichAlgorithm) bgs = new WeightedMovingMeanBGS;
	if (4 == whichAlgorithm) bgs = new WeightedMovingVarianceBGS;
	if (5 == whichAlgorithm) bgs = new MixtureOfGaussianV1BGS;
	if (6 == whichAlgorithm) bgs = new MixtureOfGaussianV2BGS;
	if (7 == whichAlgorithm) bgs = new AdaptiveBackgroundLearning;
	if (8 == whichAlgorithm) bgs = new GMG;

	// DP Package (adapted from Donovan Parks)
	if (9 == whichAlgorithm) bgs = new DPAdaptiveMedianBGS;
	if (10 == whichAlgorithm) bgs = new DPGrimsonGMMBGS;
	if (11 == whichAlgorithm) bgs = new DPZivkovicAGMMBGS;
	if (12 == whichAlgorithm) bgs = new DPMeanBGS;
	if (13 == whichAlgorithm) bgs = new DPWrenGABGS;
	if (14 == whichAlgorithm) bgs = new DPPratiMediodBGS;
	if (15 == whichAlgorithm)
	{
		bgs = new DPEigenbackgroundBGS;
		useGrayImage = false;
	}

	// TB Package (adapted from Thierry Bouwmans)
	if (16 == whichAlgorithm) bgs = new T2FGMM_UM;
	if (17 == whichAlgorithm) bgs = new T2FGMM_UV;
	if (18 == whichAlgorithm)
	{
		bgs = new T2FMRF_UM;
		useGrayImage = false;
	}
	if (19 == whichAlgorithm)
	{
		bgs = new T2FMRF_UV;
		useGrayImage = false;
	}
	if (20 == whichAlgorithm)
	{
		bgs = new FuzzySugenoIntegral;
		useGrayImage = false;
	}
	if (21 == whichAlgorithm)
	{
		bgs = new FuzzyChoquetIntegral;
		useGrayImage = false;
	}

	// JMO Package (adapted from Jean-Marc Odobez)
	if (22 == whichAlgorithm)
	{
		bgs = new MultiLayerBGS;
		useGrayImage = false;
	}

	// PT Package (adapted from Hofmann)
	if (23 == whichAlgorithm) bgs = new PixelBasedAdaptiveSegmenter;

	// LB Package (adapted from Laurence Bender)
	if (24 == whichAlgorithm)
	{
		bgs = new LBSimpleGaussian;
		useGrayImage = false;
	}
	if (25 == whichAlgorithm)
	{
		bgs = new LBFuzzyGaussian;
		useGrayImage = false;
	}
	if (26 == whichAlgorithm)
	{
		bgs = new LBMixtureOfGaussians;
		useGrayImage = false;
	}
	if (27 == whichAlgorithm)
	{
		bgs = new LBAdaptiveSOM;
		useGrayImage = false;
	}
	if (28 == whichAlgorithm)
	{
		bgs = new LBFuzzyAdaptiveSOM;
		useGrayImage = false;
	}

	if (NULL == bgs)
	{
		std::cout << "BGS object creation error" << std::endl;
		return;
	}

	cv::Mat frame, input_image, gray_image, output_image;
	while ('q' != cv::waitKey(1))
	{
		capture >> frame;
		if (frame.empty())
		{
			std::cout << "a frame not found ..." << std::endl;
			break;
			//continue;
		}

		if (input_image.empty())
			input_image = frame.clone();
		else
			frame.copyTo(input_image);

		cv::GaussianBlur(input_image, input_image, cv::Size(7, 7), 1.5);
		cv::imshow("bgslibrary: input", input_image);
		
		if (useGrayImage)
		{
			cv::cvtColor(input_image, gray_image, CV_BGR2GRAY);
			cv::imshow("bgslibrary: gray", gray_image);

			// bgs internally shows the foreground mask image
			bgs->process(gray_image, output_image);  // default
		}
		else
		{
			// bgs internally shows the foreground mask image
			bgs->process(input_image, output_image);  // use it for JMO Package and LB Package
		}

		if (!output_image.empty())
		{
			// do something
		}
	}

	delete bgs;

	cv::destroyAllWindows();
}

}  // namespace local
}  // unnamed namespace

namespace my_bgslibrary {

}  // namespace my_bgslibrary

int bgslibrary_main(int argc, char *argv[])
{
	local::demo();
	
	return 0;
}
