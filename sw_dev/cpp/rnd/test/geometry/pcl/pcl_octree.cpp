#include <ctime>
#include <vector>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>


namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html
void spatial_partitioning_and_search_operations_with_octrees_tutorial()
{
	srand ((unsigned int)time(NULL));

	// Generate pointcloud data.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		(*cloud)[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		(*cloud)[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		(*cloud)[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}

	const float resolution = 128.0f;
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	//octree.defineBoundingBox(const double min_x_arg, const double min_y_arg, const double min_z_arg, const double max_x_arg, const double max_y_arg, const double max_z_arg);

	// Neighbors within voxel search.
	pcl::PointXYZ searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

	std::vector<int> pointIdxVec;
	if (octree.voxelSearch(searchPoint, pointIdxVec))
	{
		std::cout << "Neighbors within voxel search at (" << searchPoint.x 
			<< " " << searchPoint.y 
			<< " " << searchPoint.z << ")" 
			<< std::endl;
				
		for (std::size_t i = 0; i < pointIdxVec.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxVec[i]].x 
				<< " " << (*cloud)[pointIdxVec[i]].y 
				<< " " << (*cloud)[pointIdxVec[i]].z << std::endl;
	}

	// K nearest neighbor search.
	const int K = 10;

	std::cout << "K nearest neighbor search at (" << searchPoint.x 
		<< " " << searchPoint.y 
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;

	std::vector<int> pointIdxNKNSearch;
	std::vector<float> pointNKNSquaredDistance;
	if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
			std::cout << "    "  << (*cloud)[ pointIdxNKNSearch[i]].x 
				<< " " << (*cloud)[ pointIdxNKNSearch[i]].y 
				<< " " << (*cloud)[ pointIdxNKNSearch[i]].z 
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
	}

	// Neighbors within radius search.
	const float radius = 256.0f * rand() / (RAND_MAX + 1.0f);

	std::cout << "Neighbors within radius search at (" << searchPoint.x 
		<< " " << searchPoint.y 
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	if (octree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			std::cout << "    "  << (*cloud)[ pointIdxRadiusSearch[i]].x 
				<< " " << (*cloud)[ pointIdxRadiusSearch[i]].y 
				<< " " << (*cloud)[ pointIdxRadiusSearch[i]].z 
				<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/compression.html
void point_cloud_compression_tutorial()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void octree()
{
	local::spatial_partitioning_and_search_operations_with_octrees_tutorial();
	//local::point_cloud_compression_tutorial();  // Not yet implemented.
}

}  // namespace my_pcl
