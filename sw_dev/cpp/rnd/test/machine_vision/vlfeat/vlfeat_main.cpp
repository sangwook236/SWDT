//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void sift();  // scale invariant feature transform (SIFT).
void dense_sift();
void mser();  // maximally stable extremal regions (MSER).
void hog();  // histogram of oriented gradients (HOG).
void lbp();  // local binary patterns (LBP).
void covariant_feature_detectors();
void fisher_vector();  // Fisher vector (FV) encoding.
void vlad();  // vector of locally aggregated descriptors (VLAD) encoding.
void liop();  // local intensity order pattern (LIOP) descriptor.

void kmeans();  // k-means.
void gmm();  // Gaussian mixture models (GMM).
void ikm();  // integer K-means (IKM).
void hikm();  // hierarchical Integer K-means (HIKM).
void aib();  // agglomerative information bottleneck (AIB).

void slic();  // simple linear iterative clustering (SLIC).
void quick_shift();

}  // namespace my_vlfeat

int vlfeat_main(int argc, char *argv[])
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

		// image operation ------------------------------------------
		//	image convolution.
		//	integral image.
		//	distance transform (DT).
		//	image smoothing.
		//	image gradient.
		//	affine warping.
		//	thin-plate spline warping.
		//	inverse thin-plate spline warping.

		// machine learning -----------------------------------------
		//	support vector machine (SVM).

		// feature analysis -----------------------------------------
		//	scale Invariant feature transform (SIFT).
		//	dense scale invariant feature transform (DSIFT).
		//	pyramid histogram of visual words (PHOW).
		//	maximally stable extremal regions (MSER).
		//	histogram of oriented gradients (HOG).
		//	local binary patterns (LBP).
		//	covariant feature detectors.
		//	Gaussian scale space (GSS).
		//	Fisher vector (FV) encoding.
		//	vector of locally aggregated descriptors (VLAD) encoding.
		//	local intensity order pattern (LIOP) descriptor.
		{
			//my_vlfeat::sift();
			//my_vlfeat::dense_sift();  // not yet implemented.
			//my_vlfeat::mser();
			//my_vlfeat::hog();
			my_vlfeat::lbp();
			//my_vlfeat::covariant_feature_detectors();  // not yet implemented.
			//my_vlfeat::fisher_vector();  // [ref] gmm().
			//my_vlfeat::vlad();  // [ref] gmm().
			//my_vlfeat::liop();
		}

		// clustering -----------------------------------------------
		//	k-means clustering.
		//	Gaussian mixture models (GMM).
		//	integer k-means (IKM).
		//	hierarchical integer k-means (HIKM).
		//	agglomerative information bottleneck (AIB).
		//	forests of kd-trees.
		{
			//my_vlfeat::kmeans();
			//my_vlfeat::gmm();
			//my_vlfeat::ikm();
			//my_vlfeat::hikm();
			//my_vlfeat::aib();
		}

		// segmentation ---------------------------------------------
		//	simple linear iterative clustering (SLIC).
		//	quick shift image segmentation.
		{
			//my_vlfeat::slic();
			//my_vlfeat::quick_shift();
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
