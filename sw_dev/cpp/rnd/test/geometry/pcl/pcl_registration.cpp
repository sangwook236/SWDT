#include <chrono>
#include <thread>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std::literals::chrono_literals;


namespace
{
namespace local
{

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html#iterative-closest-point
void registration_tutorial()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>(5, 1));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

	// Fill in the CloudIn data.
	for (auto &point: *cloud_in)
	{
		point.x = 1024 * rand() / (RAND_MAX + 1.0f);
		point.y = 1024 * rand() / (RAND_MAX + 1.0f);
		point.z = 1024 * rand() / (RAND_MAX + 1.0f);
	}

	std::cout << "Saved " << cloud_in->size() << " data points to input:" << std::endl;

	for (const auto &point: *cloud_in)
		std::cout << point << std::endl;

	*cloud_out = *cloud_in;

	std::cout << "Size: " << cloud_out->size() << std::endl;
	for (auto &point: *cloud_out)
		point.x += 0.7f;

	//--------------------
	std::cout << "Transformed " << cloud_in->size() << " data points:" << std::endl;

	for (const auto &point: *cloud_out)
		std::cout << point << std::endl;

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_out);

	pcl::PointCloud<pcl::PointXYZ> src_registered;
	icp.align(src_registered);

	std::cout << "Converged: " << icp.hasConverged() << ", score: " << icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;
}

void registration_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);

	// Load files.
#if 1
	const std::string input_filename1("./sample_1.pcd");
	const std::string input_filename2("./sample_2.pcd");

	const int retval1 = pcl::io::loadPCDFile<pcl::PointXYZ>(input_filename1, *cloud1);
	const int retval2 = pcl::io::loadPCDFile<pcl::PointXYZ>(input_filename2, *cloud2);
	//pcl::PCDReader reader;
	//const int retval1 = reader.read(input_filename1, *cloud1);
	//const int retval2 = reader.read(input_filename2, *cloud2);
#else
	const std::string input_filename1("./sample_1.ply");
	const std::string input_filename2("./sample_2.ply");

	const int retval1 = pcl::io::loadPLYFile<pcl::PointXYZ>(input_filename1, *cloud1);
	const int retval2 = pcl::io::loadPLYFile<pcl::PointXYZ>(input_filename2, *cloud2);
	//pcl::PLYReader reader;
	//const int retval1 = reader.read(input_filename1, *cloud1);
	//const int retval2 = reader.read(input_filename2, *cloud2);
#endif
	if (retval1 == -1)
	{
		const std::string err("File not found, " + input_filename1 + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}
	if (retval2 == -1)
	{
		const std::string err("File not found, " + input_filename2 + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	std::cout << "Loaded " << cloud1->size() << " data points from " << input_filename1 << std::endl;
	std::cout << "Loaded " << cloud2->size() << " data points from " << input_filename2 << std::endl;

	//--------------------
#if 0
	// Downsample.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud1);
	sor.setLeafSize(10.0f, 10.0f, 10.0f);
	sor.filter(*cloud1_filtered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_filtered = cloud1;
#endif

#if 0
	// Transform the source point cloud.
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	const float theta = M_PI / 8.0f;
	const float cos_theta = std::cos(theta), sin_theta = std::sin(theta);
	transform(0, 0) = cos_theta;
	transform(0, 1) = -sin_theta;
	transform(1, 0) = sin_theta;
	transform(1, 1) = cos_theta;
	transform(0, 3) = 100.0f;
	transform(1, 3) = 50.0f;
	transform(2, 3) = -50.0f;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_transformed(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*cloud1_filtered, *cloud1_transformed, transform);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_transformed = cloud1_filtered;
#endif

	//--------------------
#if 1
	// ICP.
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud1_transformed);
	icp.setInputTarget(cloud2);

	// Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored).
	//icp.setMaxCorrespondenceDistance(0.05);
	// Set the maximum number of iterations (criterion 1).
	icp.setMaximumIterations(100);
	// Set the transformation epsilon (criterion 2).
	//icp.setTransformationEpsilon(1e-8);
	// Set the Euclidean distance difference epsilon (criterion 3).
	//icp.setEuclideanFitnessEpsilon(1);

	const auto start_time(std::chrono::high_resolution_clock::now());
	pcl::PointCloud<pcl::PointXYZ> src_registered;
	icp.align(src_registered);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Point cloud registered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	std::cout << "Converged: " << icp.hasConverged() << ", score: " << icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_proxy(&src_registered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_proxy = cloud1_transformed;
#endif

	//--------------------
	// Estimate normals.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal1(new pcl::PointCloud<pcl::Normal>);
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud1_proxy);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*cloud_normal1);  // Compute the features.
	}
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal2(new pcl::PointCloud<pcl::Normal>);
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud2);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*cloud_normal2);  // Compute the features.
	}

	//--------------------
	// RGB.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb1(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_rgb1->width = cloud1_proxy->width;
	//cloud_rgb1->height = cloud1_proxy->height;
	{
		uint8_t r(127), g(0), b(0);
		for (const auto &pt: *cloud1_proxy)
		{
			pcl::PointXYZRGB point;
			point.x = pt.x;
			point.y = pt.y;
			point.z = pt.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float *>(&rgb);
			cloud_rgb1->push_back(point);
		}
	}
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb2(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_rgb2->width = cloud2->width;
	//cloud_rgb2->height = cloud2->height;
	{
		uint8_t r(0), g(0), b(127);
		for (const auto &pt: *cloud2)
		{
			pcl::PointXYZRGB point;
			point.x = pt.x;
			point.y = pt.y;
			point.z = pt.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float *>(&rgb);
			cloud_rgb2->push_back(point);
		}
	}

	// Visualize.
#if 1
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	//viewer->addPointCloud(cloud1_proxy, "point cloud 1");
	//viewer->addPointCloud(cloud2, "point cloud 2");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(cloud_rgb1);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_rgb1, rgb1, "point cloud 1");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(cloud_rgb2);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_rgb2, rgb2, "point cloud 2");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud 1");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point cloud 2");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud_rgb1, cloud_normal1, 10, 0.05, "normal 1");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud_rgb2, cloud_normal2, 10, 0.05, "normal 2");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	/*
	viewer->setCameraPosition(
		0, 30, 0,  // The coordinates of the camera location.
		0, 0, 0,  // The components of the view point of the camera.
		0, 0, 1  // The component of the view up direction of the camera.
	);
	viewer->setCameraFieldOfView(0.523599);  // [rad].
	viewer->setCameraClipDistances(0.00522511, 50);
	*/

	// Main loop.
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(10);
		std::this_thread::sleep_for(10ms);
	}
#else
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	viewer.showCloud(cloud_rgb1, "point cloud 1");
	viewer.showCloud(cloud_rgb2, "point cloud 2");
	while (!viewer.wasStopped());
#endif
}

} // namespace local
} // unnamed namespace

namespace my_pcl
{

void registration()
{
	//local::registration_tutorial();
	local::registration_example();
}

} // namespace my_pcl
