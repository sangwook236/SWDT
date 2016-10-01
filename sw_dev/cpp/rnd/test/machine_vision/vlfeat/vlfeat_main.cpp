//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void sift();  // Scale invariant feature transform (SIFT).
void dense_sift();
void mser();  // Maximally stable extremal regions (MSER).
void hog();  // Histogram of oriented gradients (HOG).
void lbp();  // Local binary patterns (LBP).
void covariant_feature_detectors();
void fisher_vector();  // Fisher vector (FV) encoding.
void vlad();  // Vector of locally aggregated descriptors (VLAD) encoding.
void liop();  // Local intensity order pattern (LIOP) descriptor.

void kmeans();  // k-means.
void gmm();  // Gaussian mixture models (GMM).
void ikm();  // Integer K-means (IKM).
void hikm();  // Hierarchical Integer K-means (HIKM).
void aib();  // Agglomerative information bottleneck (AIB).

void slic();  // Simple linear iterative clustering (SLIC).
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

		// Image operation ------------------------------------------
		//	Image convolution.
		//	Integral image.
		//	Distance transform (DT).
		//	Image smoothing.
		//	Image gradient.
		//	Affine warping.
		//	Thin-plate spline warping.
		//	Inverse thin-plate spline warping.

		// Machine learning -----------------------------------------
		//	Support vector machine (SVM).

		// Feature analysis -----------------------------------------
		//	Scale Invariant feature transform (SIFT).
		//	Dense scale invariant feature transform (DSIFT).
		//	Pyramid histogram of visual words (PHOW).
		//	Maximally stable extremal regions (MSER).
		//	Histogram of oriented gradients (HOG).
		//	Local binary patterns (LBP).
		//	Covariant feature detectors.
		//	Gaussian scale space (GSS).
		//	Fisher vector (FV) encoding.
		//	Vector of locally aggregated descriptors (VLAD) encoding.
		//	Local intensity order pattern (LIOP) descriptor.
		{
			//my_vlfeat::sift();
			//my_vlfeat::dense_sift();  // Not yet implemented.
			//my_vlfeat::mser();
			//my_vlfeat::hog();
			my_vlfeat::lbp();
			//my_vlfeat::covariant_feature_detectors();  // Not yet implemented.
			//my_vlfeat::fisher_vector();  // REF [function] >> gmm().
			//my_vlfeat::vlad();  // REF [funciton] >> gmm().
			//my_vlfeat::liop();
		}

		// Clustering -----------------------------------------------
		//	k-means clustering.
		//	Gaussian mixture models (GMM).
		//	Integer k-means (IKM).
		//	Hierarchical integer k-means (HIKM).
		//	Agglomerative information bottleneck (AIB).
		//	Forests of kd-trees.
		{
			//my_vlfeat::kmeans();
			//my_vlfeat::gmm();
			//my_vlfeat::ikm();
			//my_vlfeat::hikm();
			//my_vlfeat::aib();
		}

		// Segmentation ---------------------------------------------
		//	Simple linear iterative clustering (SLIC).
		//	Quick shift image segmentation.
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
