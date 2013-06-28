//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void sift();  // scale invariant feature transform (SIFT)
void dense_sift();
void mser();  // maximally stable extremal regions (MSER)
void hog();  // histogram of oriented gradients (HOG) features
void covariant_feature_detectors();

void kmeans();  // k-means
void ikm();  // integer K-means (IKM)
void hikm();  // hierarchical Integer K-means (HIKM)
void aib();  // agglomerative information bottleneck (AIB)

void slic();  // simple linear iterative clustering (SLIC)
void quick_shift();

}  // namespace my_vlfeat

int vlfeat_main(int argc, char *argv[])
{
	try
	{
		cv::theRNG();

		// feature analysis -----------------------------------
		{
			//my_vlfeat::sift();
			//my_vlfeat::dense_sift();  // not yet implemented
			//my_vlfeat::mser();
			//my_vlfeat::hog();  // not yet implemented
			//my_vlfeat::covariant_feature_detectors();  // not yet implemented
		}

		// clustering -----------------------------------------
		{
			//my_vlfeat::kmeans();
			//my_vlfeat::ikm();
			//my_vlfeat::hikm();
			//my_vlfeat::aib();
		}

		// segmentation ---------------------------------------
		{
			// simple linear iterative clustering (SLIC)
			//my_vlfeat::slic();

			my_vlfeat::quick_shift();
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
