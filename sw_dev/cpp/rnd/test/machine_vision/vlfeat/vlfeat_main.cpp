//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


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

}  // namespace my_vlfeat

int vlfeat_main(int argc, char *argv[])
{
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
		my_vlfeat::slic();
	}

	return 0;
}
