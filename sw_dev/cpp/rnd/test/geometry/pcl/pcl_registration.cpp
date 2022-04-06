#include <chrono>
#include <thread>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/registration/ia_kfpcs.h>
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

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/test/registration/test_sac_ia.cpp
void sac_ic_test()
{
	using PointT = pcl::PointXYZ;

	const std::string source_filepath("/path/to/sample_1.pcd");
	const std::string source_filepath("/path/to/sample_2.pcd");

	// Load the source and target cloud PCD files.
	pcl::PointCloud<PointT> cloud_source, cloud_target;
	{
		if (-1 == pcl::io::loadPCDFile(source_filepath, cloud_source))
		{
			const std::string err("File not found, " + source_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		if (-1 == pcl::io::loadPCDFile(target_filepath, cloud_target))
		{
			const std::string err("File not found, " + target_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << "Loaded " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ") from " << source_filepath << std::endl;
		std::cout << "Loaded " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ") from " << target_filepath << std::endl;

		// Trim the point clouds.
		const float depth_limit = 1.0f;
		pcl::PassThrough<PointT> pass;
		pass.setFilterFieldName("z");
		pass.setFilterLimits(0, depth_limit);
		//pass.setFilterLimits(depth_limit, std::numeric_limits<float>::max());

		pass.setInputCloud(cloud_source.makeShared());
		pass.filter(cloud_source);

		pass.setInputCloud(cloud_target.makeShared());
		pass.filter(cloud_target);

		// Downsample.
		const float voxel_size = 0.05f;
		pcl::VoxelGrid<PointT> sor;
		sor.setLeafSize(voxel_size, voxel_size, voxel_size);

		sor.setInputCloud(cloud_source.makeShared());
		sor.filter(cloud_source);

		sor.setInputCloud(cloud_target.makeShared());
		sor.filter(cloud_target);

		std::cout << "Preprocessed " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ")" << std::endl;
		std::cout << "Preprocessed " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ")" << std::endl;
	}

	// Create shared pointers.
	pcl::PointCloud<PointT>::Ptr cloud_source_ptr = cloud_source.makeShared();
	pcl::PointCloud<PointT>::Ptr cloud_target_ptr = cloud_target.makeShared();

	// SAC-IA.
	pcl::PointCloud<PointT> cloud_reg;
	pcl::SampleConsensusInitialAlignment<PointT, PointT, pcl::FPFHSignature33> sac_ia;
	{
		std::cout << "SAC-IA..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());

		// Initialize estimators for surface normals and FPFH features.
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

		pcl::NormalEstimation<PointT, pcl::Normal> norm_est;
		norm_est.setSearchMethod(tree);
		norm_est.setRadiusSearch(0.05);
		pcl::PointCloud<pcl::Normal> normals;

		pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
		fpfh_est.setSearchMethod(tree);
		fpfh_est.setRadiusSearch(0.05);

		// Estimate the FPFH features for the source cloud.
		pcl::PointCloud<pcl::FPFHSignature33> features_source;
		norm_est.setInputCloud(cloud_source_ptr);
		norm_est.compute(normals);
		fpfh_est.setInputCloud(cloud_source_ptr);
		fpfh_est.setInputNormals(normals.makeShared());
		fpfh_est.compute(features_source);

		// Estimate the FPFH features for the target cloud.
		pcl::PointCloud<pcl::FPFHSignature33> features_target;
		norm_est.setInputCloud(cloud_target_ptr);
		norm_est.compute(normals);
		fpfh_est.setInputCloud(cloud_target_ptr);
		fpfh_est.setInputNormals(normals.makeShared());
		fpfh_est.compute(features_target);

		// Initialize Sample Consensus Initial Alignment (SAC-IA).
		sac_ia.setMinSampleDistance(0.05f);
		sac_ia.setMaxCorrespondenceDistance(0.1);
		sac_ia.setMaximumIterations(1000);

		sac_ia.setInputSource(cloud_source_ptr);
		sac_ia.setInputTarget(cloud_target_ptr);
		sac_ia.setSourceFeatures(features_source.makeShared());
		sac_ia.setTargetFeatures(features_target.makeShared());

		// Register.
		sac_ia.align(cloud_reg);
		//EXPECT_EQ(cloud_reg.size(), cloud_source.size());
		//EXPECT_LT(sac_ia.getFitnessScore(), 0.0005);

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "SAC-IA performed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		std::cout << "Converged: " << sac_ia.hasConverged() << ", score: " << sac_ia.getFitnessScore() << std::endl;
		std::cout << sac_ia.getFinalTransformation() << std::endl;
	}

	// Check again, for all possible caching schemes.
	for (int iter = 0; iter < 4; ++iter)
	{
		std::cout << "SAC-IA..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());

		const bool force_cache = static_cast<bool>(iter / 2);
		const bool force_cache_reciprocal = static_cast<bool>(iter % 2);

		// Ensure that, when force_cache is not set, we are robust to the wrong input.
		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
		if (force_cache)
			tree->setInputCloud(cloud_target_ptr);
		sac_ia.setSearchMethodTarget(tree, force_cache);

		pcl::search::KdTree<PointT>::Ptr tree_recip(new pcl::search::KdTree<PointT>);
		if (force_cache_reciprocal)
			tree_recip->setInputCloud(cloud_source_ptr);
		sac_ia.setSearchMethodSource(tree_recip, force_cache_reciprocal);

		// Register.
		sac_ia.align(cloud_reg);
		//EXPECT_EQ(cloud_reg.size(), cloud_source.size());
		//EXPECT_LT(sac_ia.getFitnessScore(), 0.0005);

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "SAC-IA performed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		std::cout << "Converged: " << sac_ia.hasConverged() << ", score: " << sac_ia.getFitnessScore() << std::endl;
		std::cout << sac_ia.getFinalTransformation() << std::endl;
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/test/registration/test_fpcs_ia.cpp
void fpcs_test()
{
	using PointT = pcl::PointXYZ;

	const std::string source_filepath("/path/to/sample_1.pcd");
	const std::string source_filepath("/path/to/sample_2.pcd");

	// Load the source and target cloud PCD files.
	pcl::PointCloud<PointT> cloud_source, cloud_target;
	{
		if (-1 == pcl::io::loadPCDFile(source_filepath, cloud_source))
		{
			const std::string err("File not found, " + source_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		if (-1 == pcl::io::loadPCDFile(target_filepath, cloud_target))
		{
			const std::string err("File not found, " + target_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << "Loaded " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ") from " << source_filepath << std::endl;
		std::cout << "Loaded " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ") from " << target_filepath << std::endl;

		// Trim the point clouds.
		const float depth_limit = 1.0f;
		pcl::PassThrough<PointT> pass;
		pass.setFilterFieldName("z");
		pass.setFilterLimits(0, depth_limit);
		//pass.setFilterLimits(depth_limit, std::numeric_limits<float>::max());

		pass.setInputCloud(cloud_source.makeShared());
		pass.filter(cloud_source);

		pass.setInputCloud(cloud_target.makeShared());
		pass.filter(cloud_target);

		// Downsample.
		const float voxel_size = 0.05f;
		pcl::VoxelGrid<PointT> sor;
		sor.setLeafSize(voxel_size, voxel_size, voxel_size);

		sor.setInputCloud(cloud_source.makeShared());
		sor.filter(cloud_source);

		sor.setInputCloud(cloud_target.makeShared());
		sor.filter(cloud_target);

		std::cout << "Preprocessed " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ")" << std::endl;
		std::cout << "Preprocessed " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ")" << std::endl;
	}

	// Create shared pointers.
	pcl::PointCloud<PointT>::Ptr cloud_source_ptr = cloud_source.makeShared();
	pcl::PointCloud<PointT>::Ptr cloud_target_ptr = cloud_target.makeShared();

	// 4PCS.
	pcl::PointCloud<PointT> cloud_reg;
	pcl::registration::FPCSInitialAlignment<PointT, PointT> fpcs_ia;
	{
		std::cout << "4PCS..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());

		// Initialize 4PCS.
		const int nr_threads = 1;
		const float approx_overlap = 0.9f;
		const float delta = 1.0f;
		const int nr_samples = 100;

		fpcs_ia.setInputSource(cloud_source_ptr);
		fpcs_ia.setInputTarget(cloud_target_ptr);

		fpcs_ia.setNumberOfThreads(nr_threads);
		fpcs_ia.setApproxOverlap(approx_overlap);
		fpcs_ia.setDelta(delta, true);
		fpcs_ia.setNumberOfSamples(nr_samples);

		// Align.
		fpcs_ia.align(cloud_reg);
		//EXPECT_EQ(cloud_reg.size(), cloud_source.size());

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "4PCS performed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		std::cout << "Converged: " << fpcs_ia.hasConverged() << ", score: " << fpcs_ia.getFitnessScore() << std::endl;
		std::cout << fpcs_ia.getFinalTransformation() << std::endl;
	}

	// Check for correct coarse transformation marix.
	//const Eigen::Matrix4f transform_res_from_fpcs = fpcs_ia.getFinalTransformation();
	//for (int i = 0; i < 4; ++i)
	//	for (int j = 0; j < 4; ++j)
	//		EXPECT_NEAR(transform_res_from_fpcs(i, j), transform_from_fpcs[i][j], 0.5);
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/test/registration/test_kfpcs_ia.cpp
void kfpcs_test()
{
	//using PointT = pcl::PointXYZI;
	using PointT = pcl::PointXYZ;

	const auto previous_verbosity_level = pcl::console::getVerbosityLevel();
	pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);

	const std::string source_filepath("/path/to/sample_1.pcd");
	const std::string source_filepath("/path/to/sample_2.pcd");

	// Load the source and target cloud PCD files.
	pcl::PointCloud<PointT> cloud_source, cloud_target;
	{
		if (-1 == pcl::io::loadPCDFile(source_filepath, cloud_source))
		{
			const std::string err("File not found, " + source_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		if (-1 == pcl::io::loadPCDFile(target_filepath, cloud_target))
		{
			const std::string err("File not found, " + target_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << "Loaded " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ") from " << source_filepath << std::endl;
		std::cout << "Loaded " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ") from " << target_filepath << std::endl;

		// Trim the point clouds.
		const float depth_limit = 1.0f;
		pcl::PassThrough<PointT> pass;
		pass.setFilterFieldName("z");
		pass.setFilterLimits(0, depth_limit);
		//pass.setFilterLimits(depth_limit, std::numeric_limits<float>::max());

		pass.setInputCloud(cloud_source.makeShared());
		pass.filter(cloud_source);

		pass.setInputCloud(cloud_target.makeShared());
		pass.filter(cloud_target);

		// Downsample.
		const float voxel_size = 0.05f;
		pcl::VoxelGrid<PointT> sor;
		sor.setLeafSize(voxel_size, voxel_size, voxel_size);

		sor.setInputCloud(cloud_source.makeShared());
		sor.filter(cloud_source);

		sor.setInputCloud(cloud_target.makeShared());
		sor.filter(cloud_target);

		std::cout << "Preprocessed " << cloud_source.size() << " data points (" << pcl::getFieldsList(cloud_source) << ")" << std::endl;
		std::cout << "Preprocessed " << cloud_target.size() << " data points (" << pcl::getFieldsList(cloud_target) << ")" << std::endl;
	}

	// Create shared pointers.
	pcl::PointCloud<PointT>::Ptr cloud_source_ptr = cloud_source.makeShared();
	pcl::PointCloud<PointT>::Ptr cloud_target_ptr = cloud_target.makeShared();

	// K-4PCS.
	pcl::PointCloud<PointT> cloud_reg;
	{
		std::cout << "K-4PCS..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());

		// Initialize K-4PCS.
		const int nr_threads = 8;
		const float voxel_size = 0.1f;
		const float approx_overlap = 0.9f;
		const float abort_score = 0.0f;

		pcl::registration::KFPCSInitialAlignment<PointT, PointT> kfpcs_ia;
		kfpcs_ia.setInputSource(cloud_source_ptr);
		kfpcs_ia.setInputTarget(cloud_target_ptr);

		kfpcs_ia.setNumberOfThreads(nr_threads);
		kfpcs_ia.setApproxOverlap(approx_overlap);
		kfpcs_ia.setDelta(voxel_size, false);
		kfpcs_ia.setScoreThreshold(abort_score);

		// Repeat alignment 2 times to increase probability to ~99.99%.
		const float max_angle3d = 0.1745f, max_translation3d = 1.0f;
		float angle3d = std::numeric_limits<float>::max(), translation3d = std::numeric_limits<float>::max();
		for (int i = 0; i < 2; ++i)
		{
			kfpcs_ia.align(cloud_reg);

/*
			// Copy initial matrix.
			Eigen::Matrix4f transform_groundtruth;
			for (int ii = 0; ii < 4; ++ii)
				for (int jj = 0; jj < 4; ++jj)
					transform_groundtruth(ii, jj) = transformation_office1_office2[ii][jj];

			// Check for correct transformation.
			const Eigen::Matrix4f transform_rest = kfpcs_ia.getFinalTransformation().colPivHouseholderQr().solve(transform_groundtruth);
			angle3d = std::min(angle3d, Eigen::AngleAxisf(transform_rest.block<3, 3>(0, 0)).angle());
			translation3d = std::min(translation3d, transform_rest.block<3, 1>(0, 3).norm());
			
			if (angle3d < max_angle3d && translation3d < max_translation3d)
				break;
*/
		}

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "K-4PCS performed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		std::cout << "Converged: " << kfpcs_ia.hasConverged() << ", score: " << kfpcs_ia.getFitnessScore() << std::endl;
		std::cout << kfpcs_ia.getFinalTransformation() << std::endl;
	}

	//EXPECT_EQ(cloud_reg.size(), cloud_source.size());
	//EXPECT_NEAR(angle3d, 0.0f, max_angle3d);
	//EXPECT_NEAR(translation3d, 0.0f, max_translation3d);
	pcl::console::setVerbosityLevel(previous_verbosity_level);  // Reset verbosity level.
}

} // namespace local
} // unnamed namespace

namespace my_pcl
{

void registration()
{
	//local::registration_tutorial();
	local::registration_example();

	// Initial alignment.
	//local::sac_ic_test();  // NOTE [info] >> Slow. (~20 secs)
	//local::fpcs_test();  // NOTE [info] >> The alignment results are unstable.
	//local::kfpcs_test();  // NOTE [info] >> Too slow.
}

} // namespace my_pcl
