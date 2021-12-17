#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/octree/octree_search.h>
##include <pcl/filters/voxel_grid.h>


namespace {
namespace local {

// REF [site] >>
//	https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html
//	https://pointclouds.org/documentation/tutorials/octree.html
void spatial_partitioning_and_search_operations_with_octrees_tutorial()
{
	srand((unsigned int)time(NULL));

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
			std::cout << "    "  << (*cloud)[pointIdxNKNSearch[i]].x
				<< " " << (*cloud)[pointIdxNKNSearch[i]].y
				<< " " << (*cloud)[pointIdxNKNSearch[i]].z
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
			std::cout << "    "  << (*cloud)[pointIdxRadiusSearch[i]].x
				<< " " << (*cloud)[pointIdxRadiusSearch[i]].y
				<< " " << (*cloud)[pointIdxRadiusSearch[i]].z
				<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
}

// REF [site] >> https://pointclouds.org/documentation/tutorials/octree_change.html
void spatial_change_detection_on_unorganized_point_cloud_data_tutorial()
{
	srand((unsigned int)time(NULL));

	// Octree resolution - side length of octree voxels.
	const float resolution = 32.0f;
	// Instantiate octree-based point cloud change detection class.
	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree(resolution);

	// Generate pointcloud data for cloudA.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
	cloudA->width = 128;
	cloudA->height = 1;
	cloudA->points.resize(cloudA->width * cloudA->height);

	for (std::size_t i = 0; i < cloudA->size(); ++i)
	{
		(*cloudA)[i].x = 64.0f * rand() / (RAND_MAX + 1.0f);
		(*cloudA)[i].y = 64.0f * rand() / (RAND_MAX + 1.0f);
		(*cloudA)[i].z = 64.0f * rand() / (RAND_MAX + 1.0f);
	}

	// Add points from cloudA to octree.
	octree.setInputCloud(cloudA);
	octree.addPointsFromInputCloud();

	// Switch octree buffers: This resets octree but keeps previous tree structure in memory.
	octree.switchBuffers();

	// Generate pointcloud data for cloudB.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
	cloudB->width = 128;
	cloudB->height = 1;
	cloudB->points.resize(cloudB->width * cloudB->height);

	for (std::size_t i = 0; i < cloudB->size(); ++i)
	{
		(*cloudB)[i].x = 64.0f * rand() / (RAND_MAX + 1.0f);
		(*cloudB)[i].y = 64.0f * rand() / (RAND_MAX + 1.0f);
		(*cloudB)[i].z = 64.0f * rand() / (RAND_MAX + 1.0f);
	}

	// Add points from cloudB to octree.
	octree.setInputCloud(cloudB);
	octree.addPointsFromInputCloud();

	// Get vector of point indices from octree voxels which did not exist in previous buffer.
	std::vector<int> newPointIdxVector;
	octree.getPointIndicesFromNewVoxels(newPointIdxVector);

	// Output points.
	std::cout << "Output from getPointIndicesFromNewVoxels:" << std::endl;
	for (std::size_t i = 0; i < newPointIdxVector.size(); ++i)
		std::cout << i << "# Index:" << newPointIdxVector[i]
			<< "  Point:" << (*cloudB)[newPointIdxVector[i]].x << " "
			<< (*cloudB)[newPointIdxVector[i]].y << " "
			<< (*cloudB)[newPointIdxVector[i]].z << std::endl;
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/compression.html
void point_cloud_compression_tutorial()
{
	throw std::runtime_error("Not yet implemented");
}

void create_octree_from_point_cloud_test()
{
	const std::string input_filepath("../../../data/dental_scanner/20210924/samples_pcd/001_rgb.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		const auto start_time(std::chrono::high_resolution_clock::now());
		const int retval = pcl::io::loadPCDFile<pcl::PointXYZ>(input_filepath, *cloud);
		//pcl::PCDReader reader;
		//const int retval = reader.read(input_filename1, *cloud1);
		if (retval == -1)
		{
			const std::string err("File not found, " + input_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
		std::cout << "\tLoaded " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList(*cloud) << ") from " << input_filepath << std::endl;
	}

#if 0
	// Downsample the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>());
	{
		const float leaf_size = 5.0f;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(cloud);
		sor.setLeafSize(leaf_size, leaf_size, leaf_size);
		sor.filter(*cloud_downsampled);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point cloud downsampled: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
		std::cerr << "\tDownsampled " << cloud_downsampled->width * cloud_downsampled->height << " data points (" << pcl::getFieldsList(*cloud_downsampled) << ")." << std::endl;
	}
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled = cloud;
#endif

	const float resolution = 5.0f;
	//pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(resolution);
	pcl::octree::OctreePointCloudDensity<pcl::PointXYZ> octree(resolution);
	{
		const auto start_time(std::chrono::high_resolution_clock::now());
		octree.setInputCloud(cloud_downsampled);
		octree.addPointsFromInputCloud();
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Octree created: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		std::cout << "\tEpsilon = " << octree.getEpsilon() << std::endl;
		std::cout << "\tResolution = " << octree.getResolution() << std::endl;
		std::cout << "\tTree depth = " << octree.getTreeDepth() << std::endl;
		double min_x, min_y, min_z, max_x, max_y, max_z;
		octree.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
		std::cout << "\tBounding box: (" << min_x << ", " << min_y << ", " << min_z << "), (" << max_x << ", " << max_y << ", " << max_z << ")." << std::endl;
		std::cout << "\tVoxel density = " << octree.getVoxelDensityAtPoint(pcl::PointXYZ(-1000.0f, -1000.0f, -1000.0f)) << std::endl;

#if 0
		// Traverse.
		for (auto it = octree.begin(); it != octree.end(); ++it)
		//for (auto it = octree.breadth_begin(); it != octree.breadth_end(); ++it)
		//for (auto it = octree.depth_begin(); it != octree.depth_end(); ++it)
		//for (auto it = octree.fixed_depth_begin(); it != octree.fixed_depth_end(); ++it)
		//for (auto it = octree.leaf_breadth_begin(); it != octree.leaf_breadth_end(); ++it)
		//for (auto it = octree.leaf_depth_begin(); it != octree.leaf_depth_end(); ++it)
		{
			std::cout << "\tCurrent node: " << *it << std::endl;
			/*
			const pcl::octree::OctreeKey &key = it.getCurrentOctreeKey();
			const pcl::uindex_t depth = it.getCurrentOctreeDepth();
			const pcl::octree::OctreeNode *node = it.getCurrentOctreeNode();
			const bool is_branch = it.isBranchNode();
			const bool is_leaf = it.isLeafNode();
			const char node_config = it.getNodeConfiguration();
			const unsigned long node_id = it.getNodeID();
			it.getLeafContainer();
			it.getBranchContainer();
			*/
		}
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void octree()
{
	//local::spatial_partitioning_and_search_operations_with_octrees_tutorial();
	//local::spatial_change_detection_on_unorganized_point_cloud_data_tutorial();
	//local::point_cloud_compression_tutorial();  // Not yet implemented.

	local::create_octree_from_point_cloud_test();
}

}  // namespace my_pcl
