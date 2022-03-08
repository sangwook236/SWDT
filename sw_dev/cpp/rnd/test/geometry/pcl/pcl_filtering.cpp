#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
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
	viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, "point cloud");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void filtering()
{
	//local::downsample_point_cloud_using_voxel_grid_filter_tutorial();
	local::voxel_grid_covariance_filter_test();
}

}  // namespace my_pcl
