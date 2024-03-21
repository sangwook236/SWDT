#include <chrono>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>


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

	return 0;
}
