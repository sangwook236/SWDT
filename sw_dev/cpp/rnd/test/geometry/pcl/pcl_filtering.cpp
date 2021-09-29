#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>


namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/voxel_grid.html
void downsample_point_cloud_using_voxel_grid_filter_tutorial()
{
	const std::string filename("./table_scene_lms400.pcd");
	const std::string filename_filterd("./table_scene_lms400_downsampled.pcd");

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
	sor.filter(*cloud_filtered);

	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

	pcl::PCDWriter writer;
	writer.write(filename_filterd, *cloud_filtered, Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void filtering()
{
	local::downsample_point_cloud_using_voxel_grid_filter_tutorial();
}

}  // namespace my_pcl
