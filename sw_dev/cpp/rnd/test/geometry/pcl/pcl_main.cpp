#include <chrono>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/octree/octree_pointcloud_adjacency.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

void io_example()
{
	const std::string pcd_filepath("./test_pcd.pcd");

	// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/writing_pcd.html
	{
		pcl::PointCloud<pcl::PointXYZ> cloud;

		// Fill in the cloud data.
		cloud.width = 5;
		cloud.height = 1;
		cloud.is_dense = false;
		cloud.points.resize(cloud.width * cloud.height);

		for (auto &point: cloud)
		{
			point.x = 1024 * rand() / (RAND_MAX + 1.0f);
			point.y = 1024 * rand() / (RAND_MAX + 1.0f);
			point.z = 1024 * rand() / (RAND_MAX + 1.0f);
		}

		pcl::io::savePCDFileASCII(pcd_filepath, cloud);
		std::cout << "Saved " << cloud.size() << " data points to " << pcd_filepath << std::endl;

		for (const auto &point: cloud)
			std::cout << '\t' << point.x << ", " << point.y << ", " << point.z << std::endl;
	}

	// REF [site] >> https://pcl.readthedocs.io/en/latest/reading_pcd.html
	{
		// Load a file.
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filepath, *cloud) == -1)
		{
			const std::string err("File not found, " + pcd_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		std::cout << "Loaded "
			<< cloud->width * cloud->height
			<< " data points from " << pcd_filepath << " with the following fields:"
			<< std::endl;
		for (const auto &point: *cloud)
			std::cout << '\t' << point.x << ", " << point.y << ", " << point.z << std::endl;
	}
}

void pcd_to_ply()
{
	const std::string pcd_filepath("./input.pcd");
	const std::string ply_filepath("./output.ply");

	// Load a file.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filepath, *cloud) == -1)
	{
		pcl::console::print_error("File not found, %s.\n", pcd_filepath.c_str());
		return;
	}
	std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << pcd_filepath << std::endl;

	//--------------------
#if 0
	// Downsampling.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*cloud_filtered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = cloud;
#endif

#if 0
	// Normal estimation.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_filtered);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
	ne.compute(*normals);  // Compute the features.
#endif

#if 1
	// RGB.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_rgb->width = cloud_filtered->width;
	//cloud_rgb->height = cloud_filtered->height;
	uint8_t r(127), g(127), b(127);
	for (const auto &pt: *cloud_filtered)
	{
		pcl::PointXYZRGB point;
		point.x = pt.x;
		point.y = pt.y;
		point.z = pt.z;
		uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		point.rgb = *reinterpret_cast<float *>(&rgb);
		cloud_rgb->push_back(point);
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_proxy = cloud_rgb;
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_proxy = cloud_filtered;
#endif

	//--------------------
	// Save to a file.
	if (pcl::io::savePLYFile(ply_filepath, *cloud_proxy, false) == -1)
	//if (pcl::io::savePLYFileASCII(ply_filepath, *cloud_proxy) == -1)
	//if (pcl::io::savePLYFileBinary(ply_filepath, *cloud_proxy) == -1)
	{
		pcl::console::print_error("File not found, %s.\n", ply_filepath.c_str());
		return;
	}
	std::cout << "Saved " << cloud_proxy->width * cloud_proxy->height << " data points to " << ply_filepath << std::endl;

	//--------------------
#if 0
	// Visualization.
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	viewer.showCloud(cloud_proxy, "point cloud");
	while (!viewer.wasStopped());
#endif
}

void ply_to_pcd()
{
	const std::string ply_filepath("./input.ply");
	const std::string pcd_filepath("./output.pcd");

	// Load a file.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPLYFile<pcl::PointXYZ>(ply_filepath, *cloud) == -1)
	{
		pcl::console::print_error("File not found, %s.\n", ply_filepath.c_str());
		return;
	}
	std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << ply_filepath << std::endl;

	//--------------------
#if 0
	// Downsampling.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*cloud_filtered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = cloud;
#endif

#if 0
	// Normal estimation.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_filtered);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
	ne.compute(*normals);  // Compute the features.
#endif

#if 1
	// RGB.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_rgb->width = cloud_filtered->width;
	//cloud_rgb->height = cloud_filtered->height;
	uint8_t r(127), g(127), b(127);
	for (const auto &pt: *cloud_filtered)
	{
		pcl::PointXYZRGB point;
		point.x = pt.x;
		point.y = pt.y;
		point.z = pt.z;
		uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		point.rgb = *reinterpret_cast<float *>(&rgb);
		cloud_rgb->push_back(point);
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_proxy = cloud_rgb;
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_proxy = cloud_filtered;
#endif

	//--------------------
	// Save to a file.
	if (pcl::io::savePCDFile(pcd_filepath, *cloud_proxy, false) == -1)
	//if (pcl::io::savePCDFileASCII(pcd_filepath, *cloud_proxy) == -1)
	//if (pcl::io::savePCDFileBinary(pcd_filepath, *cloud_proxy) == -1)
	{
		pcl::console::print_error("File not found, %s.\n", pcd_filepath.c_str());
		return;
	}
	std::cout << "Saved " << cloud_proxy->width * cloud_proxy->height << " data points to " << pcd_filepath << std::endl;

	//--------------------
#if 0
	// Visualization.
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	viewer.showCloud(cloud_proxy, "point cloud");
	while (!viewer.wasStopped());
#endif
}

void basic_operation()
{
	const std::string filepath("/path/to/sample.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the file.
	{
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *cloud) == -1)
		{
			const std::string err("File not found " + filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Loaded " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 1
	{
		// Downsample.
		const auto start_time(std::chrono::high_resolution_clock::now());

		const float voxel_size = 5.0f;
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		sor.setInputCloud(cloud);
		sor.setLeafSize(voxel_size, voxel_size, voxel_size);
		//sor.setMinimumPointsNumberPerVoxel(0);
		sor.filter(*cloud);

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Filtered " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}
#endif

	{
		// Convert pcl::PointCloud<PointT>::points to C++ plain array.
		const auto start_time(std::chrono::high_resolution_clock::now());

		std::vector<std::array<float, 3>> points;
		points.reserve(cloud->size());
		for (const auto &pt: cloud->points)
			points.push_back(std::array<float, 3>{pt.x, pt.y, pt.z});

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Points converted: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		const float *arr = points[0].data();
		const float *ptr = arr;
		for (const auto &pt: cloud->points)
		{
			assert(pt.x == *ptr++);
			assert(pt.y == *ptr++);
			assert(pt.z == *ptr++);
		}
	}
}

void OctreePointCloudAdjacency_testForOcclusion_test()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		const float x_offset(0.0f), y_offset(0.0f), z_offset(0.0f);

		// For extending the bounding box
		//cloud->push_back(pcl::PointXYZ(-100.0f + x_offset, -100.0f + y_offset, -100.0f + z_offset));
		//cloud->push_back(pcl::PointXYZ(100.0f + x_offset, 100.0f + y_offset, 100.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));

		//cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 0.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));

		std::cout << "#points = " << cloud->size() << std::endl;
	}

	// Test occlusion
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_occluded(new pcl::PointCloud<pcl::PointXYZ>), cloud_visible(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Testing occlusion by pcl::octree::OctreePointCloudAdjacency..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		const pcl::PointXYZ camera_pos(0.0f, 0.0f, 0.0f);
		// NOTE [info] >> The result is sensitive to the octree resolution.
		//const double octree_resolution(1.0);  // Bad
		const double octree_resolution(3.0);
		//const double octree_resolution(10.0);  // Bad
		pcl::octree::OctreePointCloudAdjacency<pcl::PointXYZ> octree(octree_resolution);
		//octree.defineBoundingBox(const double min_x_arg, const double min_y_arg, const double min_z_arg, const double max_x_arg, const double max_y_arg, const double max_z_arg);
		octree.setInputCloud(cloud);
		octree.addPointsFromInputCloud();
		for (const auto &pt: *cloud)
		{
			if (octree.testForOcclusion(pt, camera_pos))
				cloud_occluded->push_back(pt);
			else
				cloud_visible->push_back(pt);
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Occlusion tested by pcl::octree::OctreePointCloudAdjacency (#occluded points = " << cloud_occluded->size() << ", #visible points = " << cloud_visible->size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
	}

	// Visualize
	{
#if 1
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
#if 1
		viewer.setCameraPosition(
			0.0, 0.0, 100.0,  // The coordinates of the camera location
			0.0, 0.0, 0.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 100.0);
#elif 0
		pcl::PointXYZ min_point, max_point;
		pcl::getMinMax3D(*cloud, min_point, max_point);
		std::cout << "Center point of registered point cloud = (" << (min_point.x + max_point.x) / 2 << ", " << (min_point.y + max_point.y) / 2 << ", " << (min_point.z + max_point.z) / 2 << ")." << std::endl;
		viewer.setCameraPosition(
			0.0, 0.0, 100.0,  // The coordinates of the camera location.
			(min_point.x + max_point.x) / 2.0, (min_point.y + max_point.y) / 2.0, (min_point.z + max_point.z) / 2.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 100.0);
#else
		viewer.initCameraParameters();
#endif
		viewer.setBackgroundColor(0.5, 0.5, 0.5);
		//viewer.addCoordinateSystem(10.0);

		{
			const auto &camera_center = cloud->sensor_origin_.head(3);
			viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(camera_center[0], camera_center[1], camera_center[2]), pcl::PointXYZ(0.0f, 0.0f, 0.0f), 0.0, 1.0, 0.0);

			const Eigen::Affine3f transform(Eigen::Translation3f(camera_center) * cloud->sensor_orientation_);
			viewer.addCoordinateSystem(5.0, transform, "Camera Frame");

			//viewer.addSphere(pcl::PointXYZ(0.0f, 0.0f, 10.0f), 1, 1, 1, 0, "Point #1");
			//viewer.addSphere(pcl::PointXYZ(0.0f, 0.0f, 20.0f), 1, 0, 1, 1, "Point #2");
		}

		{
			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cloud, rgb, "Point Cloud");
			viewer.addPointCloud<pcl::PointXYZ>(cloud, "Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "Point Cloud");

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb_occluded(cloud_occluded);
			//viewer.addPointCloud<pcl::PointXYZ>(cloud_occluded, rgb_occluded, "Occluded Point Cloud");
			viewer.addPointCloud<pcl::PointXYZ>(cloud_occluded, "Occluded Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6.0, "Occluded Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "Occluded Point Cloud");
		}

		while (!viewer.wasStopped())
		{
			viewer.spinOnce(1);
			//std::this_thread::sleep_for(1ms);
		}
#elif 0
		pcl::visualization::CloudViewer viewer("Simple 3D Viewer");
		viewer.showCloud(cloud, "Point Cloud");
		while (!viewer.wasStopped());
#else
		// Visualize nothing
#endif
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/voxel_grid_occlusion_estimation.cpp
void VoxelGridOcclusionEstimation_test()
{
#if defined(__SSE__)
	std::cout << "SSE found." << std::endl;
#endif
#if defined(__SSE2__)
	std::cout << "SSE2 found." << std::endl;
#endif
#if defined(__AVX__)
	std::cout << "AVX found." << std::endl;
#endif
#if defined(__AVX2__)
	std::cout << "AVX2 found." << std::endl;
#endif

#if defined(__AVX__) || defined(__AVX2__)
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		// NOTE [caution] >>
		//	When the sensor origin is not inside the bounding box of the cloud, pcl::VoxelGridOcclusionEstimation doesn't work consistently (?)
		//	pcl::PointCloud<PointT>::sensor_origin_ affects the positions at which points in its point cloud are visualized

		//cloud->sensor_origin_ = Eigen::Vector4f::Zero();  // TODO [check] >> the fourth coordinate is zero? => It doesn't matter
		//cloud->sensor_origin_ = Eigen::Vector4f(0.0f, 0.0f, 30.0f, 0.0f);  // TODO [check] >> the fourth coordinate is zero? => It doesn't matter
		//cloud->sensor_origin_ = Eigen::Vector4f(0.0f, 0.0f, -30.0f, 0.0f);  // TODO [check] >> the fourth coordinate is zero? => It doesn't matter
		//cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();  // NOTE [caution] >> Don't care
		//cloud->sensor_orientation_ = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI, Eigen::Vector3f(1.0f, 0.0f, 0.0f)));  // NOTE [caution] >> Don't care

		// NOTE [info] >> conduct tests while changing the z offset from -125 to 125, {-125, -25, -15, -5, 5, 15, 25, 125 }
		const float x_offset(0.0f), y_offset(0.0f), z_offset(-25.0f);

		// For extending the bounding box
		cloud->push_back(pcl::PointXYZ(-100.0f + x_offset, -100.0f + y_offset, -100.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(100.0f + x_offset, 100.0f + y_offset, 100.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 0.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 10.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 20.0f + y_offset, -20.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, -10.0f + z_offset));

		//cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 0.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, 10.0f + z_offset));

		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, -10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 0.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 10.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-20.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(-10.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(0.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(10.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));
		cloud->push_back(pcl::PointXYZ(20.0f + x_offset, 20.0f + y_offset, 20.0f + z_offset));

		std::cout << "#points = " << cloud->size() << std::endl;
		std::cout << "Sensor origin = " << cloud->sensor_origin_.transpose() << std::endl;
		std::cout << "Sensor orientation:\n" << cloud->sensor_orientation_.toRotationMatrix() << std::endl;
	}

	// Test occlusion
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_occluded(new pcl::PointCloud<pcl::PointXYZ>), cloud_visible(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Testing occlusion by pcl::VoxelGridOcclusionEstimation..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());

		// NOTE [info] >> The pcl::VoxelGridOcclusionEstimation class works but the leaf size is very important.
		const float leaf_size = 1.0f;
		//const float leaf_size = 5.0f;  // Bad
		pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> vg;
		vg.setInputCloud(cloud);
		vg.setLeafSize(leaf_size, leaf_size, leaf_size);
		vg.initializeVoxelGrid();

		// NOTE [info] >> AVX or AVX2 must be set.
#if 0
		std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> occluded_voxels;
		if (vg.occlusionEstimationAll(occluded_voxels))
		{
			std::cerr << "Occlusion estimation failed." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Occlusion tested by pcl::VoxelGridOcclusionEstimation (#occluded voxels = " << occluded_voxels.size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
		std::cout << "#points filtered = " << vg.getFilteredPointCloud().size() << std::endl;  // TODO [check] >> pcl::VoxelGridOcclusionEstimation::getFilteredPointCloud() doesn't work (?)
		std::cout << "Min bbox coordinates = " << vg.getMinBoundCoordinates().transpose() << ", max bbox coordinates = " << vg.getMaxBoundCoordinates().transpose() << std::endl;

#if 1
		for (const auto &voxel_coords: occluded_voxels)
		{
			const auto &voxel_center = vg.getCentroidCoordinate(voxel_coords);
			cloud_occluded->push_back(pcl::PointXYZ(voxel_center.x(), voxel_center.y(), voxel_center.z()));
		}
#else
		for (const auto &pt: *cloud)
		{
			// NOTE [caution] >> estimated voxel coordinates are a little bit strange
			//const auto &voxel_coords = vg.getGridCoordinates(pt.x, pt.y, pt.z);
			auto voxel_coords = vg.getGridCoordinates(pt.x, pt.y, pt.z);
			voxel_coords.z() -= 1;  // TODO [check] >>
			if (std::find(occluded_voxels.begin(), occluded_voxels.end(), voxel_coords) != std::end(occluded_voxels))
				cloud_occluded->push_back(pt);
			else
				cloud_visible->push_back(pt);
		}
		std::cout << "#occluded points = " << cloud_occluded->size() << ", #visible points = " << cloud_visible->size() << std::endl;
#endif

#if 0
		// For checking
		{
			const auto &voxel_coords0 = vg.getGridCoordinates(-20, -20, 10);
			std::cout << "----- (-20, -20, 10), " << voxel_coords0.transpose() << std::endl;
			const auto &voxel_coords01 = vg.getGridCoordinates(-19.5f, -20, 10);
			std::cout << "----- (-19.5, -20, 10), " << voxel_coords01.transpose() << std::endl;
			const auto &voxel_coords02 = vg.getGridCoordinates(-19, -20, 10);
			std::cout << "----- (-19, -20, 10), " << voxel_coords02.transpose() << std::endl;
			const auto &voxel_coords1 = vg.getGridCoordinates(20, 20, 21);
			std::cout << "----- (20, 20, 21), " << voxel_coords1.transpose() << std::endl;
			const auto &voxel_coords11 = vg.getGridCoordinates(20.5f, 20.5f, 21);
			std::cout << "----- (20.5, 20.5, 21), " << voxel_coords11.transpose() << std::endl;
			const auto &voxel_coords2 = vg.getGridCoordinates(21, 21, 21);
			std::cout << "----- (21, 21, 21), " << voxel_coords2.transpose() << std::endl;
			const auto &voxel_coords3 = vg.getGridCoordinates(-30, -30, -30);  // Out of bound
			std::cout << "----- (-30, -30, -30), " << voxel_coords3.transpose() << std::endl;
			const auto &voxel_coords4 = vg.getGridCoordinates(30, 30, 30);  // Out of bound
			std::cout << "----- (30, 30, 30), " << voxel_coords4.transpose() << std::endl;
		}
		for (const auto &pt: *cloud)
		{
			const auto &voxel_coords = vg.getGridCoordinates(pt.x, pt.y, pt.z);
			std::cout << "***** " << pt << ", " << voxel_coords.transpose() << std::endl;
		}
		for (const auto &voxel_coords: occluded_voxels)
		{
			std::cout << "+++++ " << voxel_coords.transpose() << std::endl;
		}
#endif
#else
		//pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);  // Verbosity off

		int voxel_state;
		for (const auto &pt: *cloud)
		{
			const auto &target_voxel = vg.getGridCoordinates(pt.x, pt.y, pt.z);
			// NOTE [info] >> The ray from the sensor origin to the center of the target voxel has to intersect with the bounding box of the input cloud
			if (0 == vg.occlusionEstimation(voxel_state, target_voxel))
			//std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> occluded_voxels;
			//if (0 == vg.occlusionEstimation(voxel_state, occluded_voxels, target_voxel))
			{
				if (voxel_state)  // Occluded(1)
					cloud_occluded->push_back(pt);
				else  // Free(0)
					cloud_visible->push_back(pt);
			}
			else
			{
				std::cerr << "Occlusion estimation failed at " << pt << std::endl;
			}
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Occlusion tested by pcl::VoxelGridOcclusionEstimation (#occluded points = " << cloud_occluded->size() << ", #visible points = " << cloud_visible->size() << "): " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count() / 1000.0f << " msecs." << std::endl;
		std::cout << "#points filtered = " << vg.getFilteredPointCloud().size() << std::endl;  // TODO [check] >> pcl::VoxelGridOcclusionEstimation::getFilteredPointCloud() doesn't work (?)
		std::cout << "Min bbox coordinates = " << vg.getMinBoundCoordinates().transpose() << ", max bbox coordinates = " << vg.getMaxBoundCoordinates().transpose() << std::endl;
#endif
	}

	// Visualize
	{
#if 1
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
#if 1
		viewer.setCameraPosition(
			0.0, 0.0, 100.0,  // The coordinates of the camera location
			0.0, 0.0, 0.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 100.0);
#elif 0
		pcl::PointXYZ min_point, max_point;
		pcl::getMinMax3D(*cloud, min_point, max_point);
		std::cout << "Center point of registered point cloud = (" << (min_point.x + max_point.x) / 2 << ", " << (min_point.y + max_point.y) / 2 << ", " << (min_point.z + max_point.z) / 2 << ")." << std::endl;
		viewer.setCameraPosition(
			0.0, 0.0, 100.0,  // The coordinates of the camera location.
			(min_point.x + max_point.x) / 2.0, (min_point.y + max_point.y) / 2.0, (min_point.z + max_point.z) / 2.0,  // The components of the view point of the camera
			0.0, 1.0, 0.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 2.0);  // [rad]
		viewer.setCameraClipDistances(1.0, 100.0);
#else
		viewer.initCameraParameters();
#endif
		viewer.setBackgroundColor(0.5, 0.5, 0.5);
		viewer.addCoordinateSystem(5.0);

		{
			const auto &camera_center = cloud->sensor_origin_.head(3);
			viewer.addLine<pcl::PointXYZ>(pcl::PointXYZ(camera_center[0], camera_center[1], camera_center[2]), pcl::PointXYZ(0.0f, 0.0f, 0.0f), 0.0, 0.5, 0.0);

			const Eigen::Affine3f transform(Eigen::Translation3f(camera_center) * cloud->sensor_orientation_);
			viewer.addCoordinateSystem(3.0, transform, "Camera Frame");
			//viewer.addCoordinateSystem(3.0, camera_center[0], camera_center[1], camera_center[2], "Camera Frame");

			//viewer.addSphere(pcl::PointXYZ(0.0f, 0.0f, 10.0f), 1, 1, 1, 0, "Point #1");
			//viewer.addSphere(pcl::PointXYZ(0.0f, 0.0f, 20.0f), 1, 0, 1, 1, "Point #2");
		}

		{
			// NOTE [caution] >> pcl::PointCloud<PointT>::sensor_origin_ affects the positions at which points in its point cloud are visualized
			cloud->sensor_origin_ = Eigen::Vector4f::Zero();
			cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
			cloud_occluded->sensor_origin_ = Eigen::Vector4f::Zero();
			cloud_occluded->sensor_orientation_ = Eigen::Quaternionf::Identity();

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cloud);
			//viewer.addPointCloud<pcl::PointXYZ>(cloud, rgb, "Point Cloud");
			viewer.addPointCloud<pcl::PointXYZ>(cloud, "Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "Point Cloud");

			//pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb_occluded(cloud_occluded);
			//viewer.addPointCloud<pcl::PointXYZ>(cloud_occluded, rgb_occluded, "Occluded Point Cloud");
			viewer.addPointCloud<pcl::PointXYZ>(cloud_occluded, "Occluded Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6.0, "Occluded Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "Occluded Point Cloud");
		}

		while (!viewer.wasStopped())
		{
			viewer.spinOnce(1);
			//std::this_thread::sleep_for(1ms);
		}
#elif 0
		pcl::visualization::CloudViewer viewer("Simple 3D Viewer");
		viewer.showCloud(cloud, "Point Cloud");
		while (!viewer.wasStopped());
#else
		// Visualize nothing
#endif
	}
#else
#error AVX or AVX2 required.
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void octree();
void resampling();
void greedy_projection();
void ransac();
void registration();
void segmentation();
void reconstruction();
void visualization(int argc, char **argv);

}  // namespace my_pcl

int pcl_main(int argc, char *argv[])
{
	// Examples:
	//	${PCL_HOME}/examples
	//	${PCL_HOME}/tools

	//local::io_example();
	//local::pcd_to_ply();
	//local::ply_to_pcd();

	local::basic_operation();

	//my_pcl::octree();
	//my_pcl::resampling();
	//my_pcl::greedy_projection();
	//my_pcl::ransac();

	//my_pcl::registration();
	//my_pcl::segmentation();
	//my_pcl::reconstruction();

	//my_pcl::visualization(argc, argv);

	//-----
	// Occlusion test
	//local::OctreePointCloudAdjacency_testForOcclusion_test();  // Fast
	//local::VoxelGridOcclusionEstimation_test();  // Stable

	//-----
	// Pose graph optimization (PGO)
	//	Refer to ${SWDT_CPP_HOME}/rnd/test/optimization/g2o/g2o_pgo_test.cpp

	// Bundle adjustment (BA)
	//	Refer to ${SWDT_CPP_HOME}/rnd/test/optimization/g2o/g2o_ba_test.cpp

	return 0;
}
