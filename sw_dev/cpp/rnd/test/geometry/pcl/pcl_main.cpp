#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


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
		std::cerr << "Saved " << cloud.size() << " data points to " << pcd_filepath << std::endl;

		for (const auto &point: cloud)
			std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;
	}

	// REF [site] >> https://pcl.readthedocs.io/en/latest/reading_pcd.html
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filepath, *cloud) == -1)  // Load a file.
		{
			PCL_ERROR("Couldn't read file test_pcd.pcd.\n");
			return;
		}
		std::cout << "Loaded "
			<< cloud->width * cloud->height
			<< " data points from test_pcd.pcd with the following fields: "
			<< std::endl;
		for (const auto &point: *cloud)
			std::cout << "    " << point.x << " " << point.y << " " << point.z << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void resampling();
void greedy_projection();
void ransac();
void registration();
void visualization(int argc, char **argv);

}  // namespace my_pcl

int pcl_main(int argc, char *argv[])
{
	// Tutorials -----------------------------------------------------------
	//	REF [site] >> http://pointclouds.org/documentation/tutorials/

	//local::io_example();

	//my_pcl::resampling();
	//my_pcl::greedy_projection();

	//my_pcl::ransac();

	my_pcl::registration();

	//my_pcl::visualization(argc, argv);

	return 0;
}
