#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/io/pcd_io.h>


namespace {
namespace local {

// REF [site] >> http://www.pointclouds.org/documentation/tutorials/resampling.php
void resampling_tutorial()
{
	// Load input file into a PointCloud<T> with an appropriate type.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	// Load bun0.pcd -- should be available with the PCL archive in test.
	pcl::io::loadPCDFile("./data/geometry/pcl/bun0.pcd", *cloud);

	// Create a KD-Tree.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	// Output has the PointNormal type in order to store the normals calculated by MLS.
	pcl::PointCloud<pcl::PointNormal> mls_points;

	// Init object (second point type is for the normals, even if unused).
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

	mls.setComputeNormals(true);

	// Set parameters.
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.03);

	// Reconstruct.
	mls.process(mls_points);

	// Save output.
	pcl::io::savePCDFile("./data/geometry/pcl/bun0_mls.pcd", mls_points);
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void resampling()
{
	local::resampling_tutorial();
}

}  // namespace my_pcl
