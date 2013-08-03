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
#include "../bgslibrary_lib/package_bgs/dp/DPTextureBGS.h"

#include "../bgslibrary_lib/package_bgs/tb/T2FGMM_UM.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FGMM_UV.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FMRF_UM.h"
#include "../bgslibrary_lib/package_bgs/tb/T2FMRF_UV.h"
#include "../bgslibrary_lib/package_bgs/tb/FuzzySugenoIntegral.h"
#include "../bgslibrary_lib/package_bgs/tb/FuzzyChoquetIntegral.h"

#include "../bgslibrary_lib/package_bgs/lb/LBSimpleGaussian.h"
#include "../bgslibrary_lib/package_bgs/lb/LBFuzzyGaussian.h"
#include "../bgslibrary_lib/package_bgs/lb/LBMixtureOfGaussians.h"
#include "../bgslibrary_lib/package_bgs/lb/LBAdaptiveSOM.h"
#include "../bgslibrary_lib/package_bgs/lb/LBFuzzyAdaptiveSOM.h"

#include "../bgslibrary_lib/package_bgs/jmo/MultiLayerBGS.h"
#include "../bgslibrary_lib/package_bgs/pt/PixelBasedAdaptiveSegmenter.h"
#include "../bgslibrary_lib/package_bgs/av/VuMeter.h"
#include "../bgslibrary_lib/package_bgs/ae/KDE.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cassert>


namespace {
namespace local {

void demo()
{
#if 1
	const std::string avi_filename("./data/change_detection/video.avi");
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

	enum BGS_ALGO {
		ALGO_FrameDifferenceBGS,
		ALGO_StaticFrameDifferenceBGS,
		ALGO_WeightedMovingMeanBGS,
		ALGO_WeightedMovingVarianceBGS,
		ALGO_MixtureOfGaussianV1BGS,
		ALGO_MixtureOfGaussianV2BGS,
		ALGO_AdaptiveBackgroundLearning,
		ALGO_GMG,

		ALGO_DPAdaptiveMedianBGS,
		ALGO_DPGrimsonGMMBGS,
		ALGO_DPZivkovicAGMMBGS,
		ALGO_DPMeanBGS,
		ALGO_DPWrenGABGS,
		ALGO_DPPratiMediodBGS,
		ALGO_DPEigenbackgroundBGS,
		ALGO_DPTextureBGS,

		ALGO_T2FGMM_UM,
		ALGO_T2FGMM_UV,
		ALGO_T2FMRF_UM,
		ALGO_T2FMRF_UV,
		ALGO_FuzzySugenoIntegral,
		ALGO_FuzzyChoquetIntegral,

		ALGO_LBSimpleGaussian,
		ALGO_LBFuzzyGaussian,
		ALGO_LBMixtureOfGaussians,
		ALGO_LBAdaptiveSOM,
		ALGO_LBFuzzyAdaptiveSOM,

		ALGO_MultiLayerBGS,
		ALGO_PixelBasedAdaptiveSegmenter,  // PBAS
		ALGO_VuMeter,
		ALGO_KDE,
	};

	const BGS_ALGO whichAlgorithm = ALGO_MixtureOfGaussianV2BGS;
	const bool applyGaussianBlurOnInputImage = false;
	const bool applyMedianFilterOnMaskImage = false;

	bool useColorImage = true;
	IBGS *bgs = NULL;

	// background subtraction methods
	switch (whichAlgorithm)
	{
	case ALGO_FrameDifferenceBGS:
		bgs = new FrameDifferenceBGS;
		break;
	case ALGO_StaticFrameDifferenceBGS:
		bgs = new StaticFrameDifferenceBGS;
		break;
	case ALGO_WeightedMovingMeanBGS:
		bgs = new WeightedMovingMeanBGS;
		break;
	case ALGO_WeightedMovingVarianceBGS:
		bgs = new WeightedMovingVarianceBGS;
		break;
	case ALGO_MixtureOfGaussianV1BGS:  // run-time error
		bgs = new MixtureOfGaussianV1BGS;
		//useColorImage = false;
		break;
	case ALGO_MixtureOfGaussianV2BGS:
		bgs = new MixtureOfGaussianV2BGS;
		break;
	case ALGO_AdaptiveBackgroundLearning:
		bgs = new AdaptiveBackgroundLearning;
		break;
	case ALGO_GMG:  // run-time error
		bgs = new GMG;
		//useColorImage = false;
		break;

	// DP Package (adapted from Donovan Parks)
	case ALGO_DPAdaptiveMedianBGS:
		bgs = new DPAdaptiveMedianBGS;
		break;
	case ALGO_DPGrimsonGMMBGS:
		bgs = new DPGrimsonGMMBGS;
		break;
	case ALGO_DPZivkovicAGMMBGS:
		bgs = new DPZivkovicAGMMBGS;
		break;
	case ALGO_DPMeanBGS:
		bgs = new DPMeanBGS;
		break;
	case ALGO_DPWrenGABGS:
		bgs = new DPWrenGABGS;
		break;
	case ALGO_DPPratiMediodBGS:
		bgs = new DPPratiMediodBGS;
		break;
	case ALGO_DPEigenbackgroundBGS:
		bgs = new DPEigenbackgroundBGS;
		break;
	case ALGO_DPTextureBGS:
		bgs = new DPTextureBGS;
		break;

	// TB Package (adapted from Thierry Bouwmans)
	case ALGO_T2FGMM_UM:
		bgs = new T2FGMM_UM;
		break;
	case ALGO_T2FGMM_UV:
		bgs = new T2FGMM_UV;
		break;
	case ALGO_T2FMRF_UM:
		bgs = new T2FMRF_UM;
		break;
	case ALGO_T2FMRF_UV:
		bgs = new T2FMRF_UV;
		break;
	case ALGO_FuzzySugenoIntegral:
		bgs = new FuzzySugenoIntegral;
		break;
	case ALGO_FuzzyChoquetIntegral:
		bgs = new FuzzyChoquetIntegral;
		break;

	// LB Package (adapted from Laurence Bender)
	case ALGO_LBSimpleGaussian:
		bgs = new LBSimpleGaussian;
		break;
	case ALGO_LBFuzzyGaussian:
		bgs = new LBFuzzyGaussian;
		break;
	case ALGO_LBMixtureOfGaussians:
		bgs = new LBMixtureOfGaussians;
		break;
	case ALGO_LBAdaptiveSOM:
		bgs = new LBAdaptiveSOM;
		break;
	case ALGO_LBFuzzyAdaptiveSOM:
		bgs = new LBFuzzyAdaptiveSOM;
		break;

	// JMO Package (adapted from Jean-Marc Odobez)
	case ALGO_MultiLayerBGS:
		bgs = new MultiLayerBGS;
		break;

	// PT Package (adapted from Hofmann)
	case ALGO_PixelBasedAdaptiveSegmenter:
		bgs = new PixelBasedAdaptiveSegmenter;
		break;

	//
	case ALGO_VuMeter:
		bgs = new VuMeter;
		break;
	case ALGO_KDE:
		bgs = new KDE;
		break;

	default:
		bgs = NULL;
		break;
	}

	if (NULL == bgs)
	{
		std::cout << "BGS object creation error" << std::endl;
		return;
	}

	cv::Mat frame, input_image, gray_image, mask_image, bkg_image;
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

		if (applyGaussianBlurOnInputImage)
			cv::GaussianBlur(input_image, input_image, cv::Size(7, 7), 1.5);

		cv::imshow("bgslibrary: input", input_image);

		if (useColorImage)
		{
			// bgs internally shows the foreground mask image
			bgs->process(input_image, mask_image, bkg_image);  // use it for JMO Package and LB Package
		}
		else
		{
			cv::cvtColor(input_image, gray_image, CV_BGR2GRAY);
			cv::imshow("bgslibrary: gray", gray_image);

			// bgs internally shows the foreground mask image
			bgs->process(gray_image, mask_image, bkg_image);
		}

		if (applyMedianFilterOnMaskImage)
			cv::medianBlur(mask_image, mask_image, 5);

		if (!mask_image.empty())
		{
			// do something
		}

		if (!bkg_image.empty())
		{
			// do something
		}
	}

	if (bgs)
	{
		delete bgs;
		bgs = NULL;
	}

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
