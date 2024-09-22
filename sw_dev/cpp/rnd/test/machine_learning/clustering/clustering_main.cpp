//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void kmeanspp();
void k_medoids();

void spectral_clustering();

}  // namespace my_clustering

int clustering_main(int argc, char *argv[])
{
	// k-means & k-means++.
	my_clustering::kmeanspp();

	//my_clustering::k_medoids();  // Not yet implemented.

	my_clustering::spectral_clustering();

	// DBSCAN.
	//	Refer to dbscan() & dbscan_gpu() in ${SWDT_CPP_RND_HOME}/test/geometry/pcl/pcl_segmentation.cpp

	return 0;
}
