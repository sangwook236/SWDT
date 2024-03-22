#include <chrono>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>


namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html
void downsample_point_cloud_using_voxel_grid_filter_tutorial()
{
	const std::string filename("../table_scene_lms400.pcd");
	const std::string filename_filtered("../table_scene_lms400_downsampled.pcd");

	pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
	pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

	// Fill in the cloud data.
	pcl::PCDReader reader;
	// Replace the path below with the path where you saved your file.
	reader.read(filename, *cloud);  // Remember to download the file first!

	std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

	// Create the filtering object.
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	//sor.setMinimumPointsNumberPerVoxel(0);
	sor.filter(*cloud_filtered);

	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

	pcl::PCDWriter writer;
	writer.write(filename_filtered, *cloud_filtered, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);
}

void voxel_grid_covariance_filter_test()
{
	const std::string input_filepath("../table_scene_lms400.pcd");
	const std::string output_filepath("../table_scene_lms400_downsampled.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());

	// Fill in the cloud data.
	pcl::PCDReader reader;
	// Replace the path below with the path where you saved your file.
	reader.read(input_filepath, *cloud);  // Remember to download the file first!

	std::cerr << "PointCloud before filtering: " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

	// Create the filtering object.
	pcl::VoxelGridCovariance<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	//sor.setMinimumPointsNumberPerVoxel(6);
	sor.setCovEigValueInflationRatio(0.01);
	sor.filter(*cloud_filtered);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_display(new pcl::PointCloud<pcl::PointXYZ>());
	sor.getDisplayCloud(*cloud_display);

	std::cerr << "PointCloud after filtering: " << cloud_filtered->size() << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;
	std::cerr << "PointCloud for display: " << cloud_display->size() << " data points (" << pcl::getFieldsList(*cloud_display) << ")." << std::endl;

	pcl::PCDWriter writer;
	writer.write(output_filepath, *cloud_filtered);
	//writer.write(output_filepath, *cloud_display);

	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud(cloud, 255, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud, handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud_filtered(cloud_filtered, 0, 0, 255);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, handler_cloud_filtered, "point cloud filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud filtered");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

#if 0
void bilateral_filter_test()
{
#if 1
	const float sigma_s = 2.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
	const float sigma_r = 0.01f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)
#else
	const float sigma_s = 20.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
	const float sigma_r = 0.1f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)
#endif

	//const std::string input_filepath("./milk_cartoon_all_small_clorox.pcd");
	const std::string input_filepath("./input.pcd");
	const std::string output_filepath("./bf_output.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Loading a point cloud from " << input_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(input_filepath, *cloud) < 0)
		pcl::PCDReader reader;
		if (reader.read(input_filepath, *cloud) != 0)
		{
			std::cerr << "A point cloud file not found, " << input_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points." << std::endl;
		std::cout << "Available dimensions: " << pcl::getFieldsList(*cloud).c_str() << std::endl;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
	{
		pcl::BilateralFilter<pcl::PointXYZ> bf;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		bf.setInputCloud(cloud);
		//bf.computePointWeight(pid, indices, distances);
		bf.setHalfSize(sigma_s);
		bf.setStdDev(sigma_r);
		bf.setSearchMethod(tree);

		std::cout << "Filtering..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		bf.filter(*cloud_filtered);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Filtered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_filtered->width * cloud_filtered->height << " points filtered." << std::endl;
	}

	{
		std::cout << "Saving the filtered point cloud to " << output_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::savePCDFile(output_filepath, *cloud_filtered) < 0)
		pcl::PCDWriter writer;
		if (writer.write(output_filepath, *cloud_filtered) != 0)
		{
			std::cerr << "The filtered point cloud not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The filtered point cloud saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 1
	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud(cloud, 255, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud, handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud_filtered(cloud_filtered, 0, 0, 255);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, handler_cloud_filtered, "point cloud filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud filtered");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}
#endif

void fast_bilateral_filter_test()
{
#if 1
	const float sigma_s = 2.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
	const float sigma_r = 0.01f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)
#else
	const float sigma_s = 20.0f;  // The standard deviation of the Gaussian used by the bilateral filter for the spatial neighborhood/window
	const float sigma_r = 0.1f;  // The standard deviation of the Gaussian used to control how much an adjacent pixel is downweighted because of the intensity difference (depth in our case)
#endif

	//const std::string input_filepath("./milk_cartoon_all_small_clorox.pcd");
	const std::string input_filepath("./input.pcd");
	const std::string output_filepath("./fbf_output.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Loading a point cloud from " << input_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(input_filepath, *cloud) < 0)
		pcl::PCDReader reader;
		if (reader.read(input_filepath, *cloud) != 0)
		{
			std::cerr << "A point cloud file not found, " << input_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud->width * cloud->height << " points." << std::endl;
		std::cout << "Available dimensions: " << pcl::getFieldsList(*cloud).c_str() << std::endl;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
	{
		pcl::FastBilateralFilter<pcl::PointXYZ> fbf;
		//pcl::FastBilateralFilterOMP<pcl::PointXYZ> fbf;
		fbf.setInputCloud(cloud);
		fbf.setSigmaS(sigma_s);
		fbf.setSigmaR(sigma_r);

		std::cout << "Filtering..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		fbf.filter(*cloud_filtered);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Filtered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
		std::cout << cloud_filtered->width * cloud_filtered->height << " points filtered." << std::endl;
	}

	{
		std::cout << "Saving the filtered point cloud to " << output_filepath << "..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::savePCDFile(output_filepath, *cloud_filtered) < 0)
		pcl::PCDWriter writer;
		if (writer.write(output_filepath, *cloud_filtered) != 0)
		{
			std::cerr << "The filtered point cloud not saved." << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "The filtered point cloud saved: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

#if 1
	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud(cloud, 255, 0, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud, handler_cloud, "point cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_cloud_filtered(cloud_filtered, 0, 0, 255);
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, handler_cloud_filtered, "point cloud filtered");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud filtered");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void filtering()
{
	//local::downsample_point_cloud_using_voxel_grid_filter_tutorial();
	//local::voxel_grid_covariance_filter_test();

	// Bilateral upsampling
	//	REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/tools/bilateral_upsampling.cpp

	// Bilateral filtering
	//local::bilateral_filter_test();  // Linking error
	local::fast_bilateral_filter_test();

	// Mean least squres (MLS) smoothing.
	//	Refer to mls_smoothing_example() in pcl_recostruction.cpp
}

}  // namespace my_pcl
