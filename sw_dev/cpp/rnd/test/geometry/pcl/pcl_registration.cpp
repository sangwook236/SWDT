#include <chrono>
#include <thread>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_estimation_organized_projection.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_organized_boundary.h>
#include <pcl/registration/correspondence_rejection_poly.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_sample_consensus_2d.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_dq.h>
#include <pcl/registration/transformation_estimation_dual_quaternion.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_symmetric_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/registration/transformation_validation.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/registration/ia_kfpcs.h>


using namespace std::literals::chrono_literals;

namespace
{
namespace local
{

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html
void icp_registration_tutorial()
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

	std::cout << cloud_in->size() << " data points saved to input:" << std::endl;

	for (const auto &point: *cloud_in)
		std::cout << point << std::endl;

	*cloud_out = *cloud_in;

	std::cout << "Size: " << cloud_out->size() << std::endl;
	for (auto &point: *cloud_out)
		point.x += 0.7f;

	//--------------------
	std::cout << cloud_in->size() << " data points transformed:" << std::endl;

	for (const auto &point: *cloud_out)
		std::cout << point << std::endl;

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_out);

	pcl::PointCloud<pcl::PointXYZ> src_registered;
	icp.align(src_registered);

	std::cout << "Converged = " << icp.hasConverged() << ", score = " << icp.getFitnessScore() << std::endl;
	std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;
}

void icp_registration_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
	{
		// Load files.
#if 1
		const std::string src_filename("/path/to/src_sample.pcd");
		const std::string tgt_filename("/path/to/tgt_sample.pcd");

		const int retval_src = pcl::io::loadPCDFile<pcl::PointXYZ>(src_filename, *cloud_src);
		const int retval_tgt = pcl::io::loadPCDFile<pcl::PointXYZ>(tgt_filename, *cloud_tgt);
		//pcl::PCDReader reader;
		//const int retval_src = reader.read(src_filename, *cloud_src);
		//const int retval_tgt = reader.read(tgt_filename, *cloud_tgt);
#else
		const std::string src_filename("/path/to/src_sample.ply");
		const std::string tgt_filename("/path/to/tgt_sample.ply");

		const int retval_src = pcl::io::loadPLYFile<pcl::PointXYZ>(src_filename, *cloud_src);
		const int retval_tgt = pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_filename, *cloud_tgt);
		//pcl::PLYReader reader;
		//const int retval_src = reader.read(src_filename, *cloud_src);
		//const int retval_tgt = reader.read(tgt_filename, *cloud_tgt);
#endif
		if (retval_src == -1)
		{
			const std::string err("File not found, " + src_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		if (retval_tgt == -1)
		{
			const std::string err("File not found, " + tgt_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << cloud_src->size() << " data points loaded from " << src_filename << std::endl;
		std::cout << cloud_tgt->size() << " data points loaded from " << tgt_filename << std::endl;
	}

	//--------------------
#if 0
	// Downsample.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_src);
	sor.setLeafSize(10.0f, 10.0f, 10.0f);
	sor.filter(*cloud_src_filtered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_filtered = cloud_src;
#endif

#if 0
	// Transform the source point cloud.
	const Eigen::Vector3f translation(100.0f, 50.0f, -50.0f);
	const float angle(M_PI / 8.0f);
	const Eigen::Vector3f axis(0, 0, 1);
	const auto transform(Eigen::Translation3f(translation) * Eigen::AngleAxisf(angle, axis));

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_transformed(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*cloud_src_filtered, *cloud_src_transformed, transform.matrix());
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_transformed = cloud_src_filtered;
#endif

	//--------------------
#if 1
	// ICP.
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//pcl::IterativeClosestPointWithNormals<pcl::PointXYZ, pcl::PointXYZ> icp;
	//pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
	//pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//pcl::GeneralizedIterativeClosestPoint6D icp;
	//pcl::JointIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_src_transformed);
	icp.setInputTarget(cloud_tgt);

	// Set the maximum distance threshold between two correspondent points in source <-> target. (e.g., correspondences with higher distances will be ignored).
	//icp.setMaxCorrespondenceDistance(0.05);  // 5cm.
	// Set the maximum number of iterations.
	icp.setMaximumIterations(100);
	// Set the number of iterations RANSAC should run for.
	//icp.setRANSACIterations(100);
	// Set the inlier distance threshold for the internal RANSAC outlier rejection loop.
	//icp.setRANSACOutlierRejectionThreshold(0.1);
	// Set the transformation epsilon (maximum allowable translation squared difference between two consecutive transformations).
	//icp.setTransformationEpsilon(1e-8);
	// Set the transformation rotation epsilon (maximum allowable rotation difference between two consecutive transformations).
	//icp.setTransformationRotationEpsilon(1e-8);
	// Set the Euclidean distance difference epsilon (maximum allowed Euclidean error between two consecutive steps).
	//icp.setEuclideanFitnessEpsilon(1);

#if 0
	pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal>::Ptr correspodence_est(new pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointXYZ, pcl::PointXYZ, pcl::Normal>());
	//correspodence_est->setVoxelRepresentationTarget(dt);  // Not supported.
	correspodence_est->setInputSource(cloud_src_transformed);
	correspodence_est->setInputTarget(cloud_tgt);
	//correspodence_est->setMaxCorrespondenceDistance(max_corr_distance);
	icp.setCorrespondenceEstimation(correspodence_est);

	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>::Ptr correspodence_rej(new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ>());
	//pcl::registration::make_shared::Ptr correspodence_rej(std::make_shared<pcl::registration::CorrespondenceRejectorOneToOne>());
	correspodence_rej->setInputSource(cloud_src_transformed);
	correspodence_rej->setInputTarget(cloud_tgt);
	//correspodence_rej->setMaximumIterations(max_iterations);
	//correspodence_rej->setInlierThreshold(inlier_threshold);
	icp.addCorrespondenceRejector(correspodence_rej);

	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Ptr transformation_est(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>());
	icp.setTransformationEstimation(transformation_est);
#endif

	const auto start_time(std::chrono::high_resolution_clock::now());
	pcl::PointCloud<pcl::PointXYZ> src_registered;
	icp.align(src_registered);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Point cloud registered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	std::cout << "Converged = " << icp.hasConverged() << ", score = " << icp.getFitnessScore() << std::endl;
	std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;

	// Correspondences.
	//icp.correspondences_

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_proxy(&src_registered);
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src_proxy = cloud_src_transformed;
#endif

	//--------------------
	// Estimate normals.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal_src(new pcl::PointCloud<pcl::Normal>);
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud_src_proxy);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*cloud_normal_src);  // Compute the features.
	}
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal_tgt(new pcl::PointCloud<pcl::Normal>);
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud_tgt);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*cloud_normal_tgt);  // Compute the features.
	}

	//--------------------
	// RGB.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_src_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_src_rgb->width = cloud_src_proxy->width;
	//cloud_src_rgb->height = cloud_src_proxy->height;
	{
		uint8_t r(127), g(0), b(0);
		for (const auto &pt: *cloud_src_proxy)
		{
			pcl::PointXYZRGB point;
			point.x = pt.x;
			point.y = pt.y;
			point.z = pt.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float *>(&rgb);
			cloud_src_rgb->push_back(point);
		}
	}
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tgt_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
	//cloud_tgt_rgb->width = cloud_tgt->width;
	//cloud_tgt_rgb->height = cloud_tgt->height;
	{
		uint8_t r(0), g(0), b(127);
		for (const auto &pt: *cloud_tgt)
		{
			pcl::PointXYZRGB point;
			point.x = pt.x;
			point.y = pt.y;
			point.z = pt.z;
			uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			point.rgb = *reinterpret_cast<float *>(&rgb);
			cloud_tgt_rgb->push_back(point);
		}
	}

	// Visualize.
#if 1
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	//viewer->addPointCloud(cloud_src_proxy, "source point cloud");
	//viewer->addPointCloud(cloud_tgt, "target point cloud");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_src(cloud_src_rgb);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_src_rgb, rgb_src, "source point cloud");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_tgt(cloud_tgt_rgb);
	viewer->addPointCloud<pcl::PointXYZRGB>(cloud_tgt_rgb, rgb_tgt, "target point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source point cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target point cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud_src_rgb, cloud_normal_src, 10, 0.05, "source normals");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud_tgt_rgb, cloud_normal_tgt, 10, 0.05, "target normals");
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
	viewer->resetCamera();

	// Main loop.
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(10);
		std::this_thread::sleep_for(10ms);
	}
#else
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	viewer.showCloud(cloud_src_rgb, "source point cloud");
	viewer.showCloud(cloud_tgt_rgb, "target point cloud");
	while (!viewer.wasStopped());
#endif
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/normal_distributions_transform.html
void ndt_example()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	{
		// Load files.
#if 1
		const std::string src_filename("/path/to/src_sample.pcd");
		const std::string tgt_filename("/path/to/tgt_sample.pcd");

		const int retval_src = pcl::io::loadPCDFile<pcl::PointXYZ>(src_filename, *cloud_source);
		const int retval_tgt = pcl::io::loadPCDFile<pcl::PointXYZ>(tgt_filename, *cloud_target);
		//pcl::PCDReader reader;
		//const int retval_src = reader.read(src_filename, *cloud_source);
		//const int retval_tgt = reader.read(tgt_filename, *cloud_target);
#else
		const std::string src_filename("/path/to/src_sample.ply");
		const std::string tgt_filename("/path/to/tgt_sample.ply");

		const int retval_src = pcl::io::loadPLYFile<pcl::PointXYZ>(src_filename, *cloud_source);
		const int retval_tgt = pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_filename, *cloud_target);
		//pcl::PLYReader reader;
		//const int retval_src = reader.read(src_filename, *cloud_source);
		//const int retval_tgt = reader.read(tgt_filename, *cloud_target);
#endif
		if (retval_src == -1)
		{
			const std::string err("File not found, " + src_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		if (retval_tgt == -1)
		{
			const std::string err("File not found, " + tgt_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << cloud_source->size() << " data points loaded from " << src_filename << std::endl;
		std::cout << cloud_target->size() << " data points loaded from " << tgt_filename << std::endl;
	}

	//--------------------
	// Filtering input scan to roughly 10% of original size to increase speed of registration.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
	approximate_voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
	approximate_voxel_filter.setInputCloud(cloud_source);
	approximate_voxel_filter.filter(*cloud_source_filtered);
	std::cout << cloud_source_filtered->size() << " data points filtered." << std::endl;

	// Set initial alignment estimate found using robot odometry.
	const Eigen::AngleAxisf init_rotation(0.6931f, Eigen::Vector3f::UnitZ());
	const Eigen::Translation3f init_translation(1.79387f, 0.720047f, 0.0f);
	const Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

	//--------------------
	// Initializing Normal Distributions Transform (NDT).
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

	// Setting scale dependent NDT parameters.
	// Setting minimum transformation difference for termination condition.
	ndt.setTransformationEpsilon(0.01);
	// Setting maximum step size for More-Thuente line search.
	ndt.setStepSize(0.1);
	//Setting Resolution of NDT grid structure (VoxelGridCovariance).
	ndt.setResolution(1.0f);

	// Setting max number of registration iterations.
	ndt.setMaximumIterations(35);

	// Setting point cloud to be aligned.
	ndt.setInputSource(cloud_source_filtered);
	// Setting point cloud to be aligned to.
	ndt.setInputTarget(cloud_target);

	// Calculating required rigid transform to align the input cloud to the target cloud.
	const auto start_time(std::chrono::high_resolution_clock::now());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_registered(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*cloud_source_registered, init_guess);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Point cloud registered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	std::cout << "Normal Distributions Transform has converged: " << ndt.hasConverged() << ", score: " << ndt.getFitnessScore() << std::endl;
	std::cout << "Transformation:\n" << ndt.getFinalTransformation() << std::endl;

	//-----
	// Transforming unfiltered, input cloud using found transform.
	pcl::transformPointCloud(*cloud_source, *cloud_source_registered, ndt.getFinalTransformation());

	//-----
	// Saving transformed input cloud.
	//pcl::io::savePCDFileASCII("/path/to/registered.pcd", *cloud_source_registered);

	//--------------------
	// Initializing point cloud visualizer.
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);

	// Coloring and visualizing target cloud (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_target, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_target, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

	// Coloring and visualizing transformed input cloud (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(cloud_source_registered, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_source_registered, output_color, "source cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

	// Starting visualizer.
	viewer->addCoordinateSystem(1.0, "global");
	viewer->initCameraParameters();
	viewer->resetCamera();

	// Wait until visualizer window is closed.
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		//std::this_thread::sleep_for(100ms);
	}
}

// REF [site] >> https://pcl.readthedocs.io/en/latest/alignment_prerejective.html
void sample_consensus_prerejective_example()
{
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_object(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointNormal>);
	{
		// Load files.
#if 1
		const std::string object_filename("/path/to/object.pcd");
		const std::string scene_filename("/path/to/scene.pcd");

		const int retval_src = pcl::io::loadPCDFile<pcl::PointNormal>(object_filename, *cloud_object);
		const int retval_tgt = pcl::io::loadPCDFile<pcl::PointNormal>(scene_filename, *cloud_scene);
		//pcl::PCDReader reader;
		//const int retval_src = reader.read(object_filename, *cloud_object);
		//const int retval_tgt = reader.read(scene_filename, *cloud_scene);
#else
		const std::string object_filename("/path/to/object.ply");
		const std::string scene_filename("/path/to/scene.ply");

		const int retval_src = pcl::io::loadPLYFile<pcl::PointNormal>(object_filename, *cloud_object);
		const int retval_tgt = pcl::io::loadPLYFile<pcl::PointNormal>(scene_filename, *cloud_scene);
		//pcl::PLYReader reader;
		//const int retval_src = reader.read(object_filename, *cloud_object);
		//const int retval_tgt = reader.read(scene_filename, *cloud_scene);
#endif
		if (retval_src == -1)
		{
			const std::string err("File not found, " + object_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		if (retval_tgt == -1)
		{
			const std::string err("File not found, " + scene_filename + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}

		std::cout << cloud_object->size() << " data points loaded from " << object_filename << std::endl;
		std::cout << cloud_scene->size() << " data points loaded from " << scene_filename << std::endl;
	}

	//--------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_scene_downsampled(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_features(new pcl::PointCloud<pcl::FPFHSignature33>);

	// Downsample.
	std::cout << "Downsampling..." << std::endl;
	pcl::VoxelGrid<pcl::PointNormal> grid;
	const float leaf = 0.005f;
	grid.setLeafSize(leaf, leaf, leaf);
	grid.setInputCloud(cloud_object);
	grid.filter(*cloud_object);
	grid.setInputCloud(cloud_scene);
	grid.filter(*cloud_scene_downsampled);

	// Estimate normals for scene.
	std::cout << "Estimating scene normals..." << std::endl;
	pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> nest;
	nest.setRadiusSearch(0.005);
	nest.setInputCloud(cloud_scene_downsampled);
	nest.setSearchSurface(cloud_scene);
	nest.compute(*cloud_scene_downsampled);

	// Estimate features.
	std::cout << "Estimating features..." << std::endl;
	pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
	fest.setRadiusSearch(0.025);
	fest.setInputCloud(cloud_object);
	fest.setInputNormals(cloud_object);
	fest.compute(*object_features);
	fest.setInputCloud(cloud_scene_downsampled);
	fest.setInputNormals(cloud_scene_downsampled);
	fest.compute(*scene_features);

	//--------------------
	// Perform alignment.
	pcl::SampleConsensusPrerejective<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> align;
	align.setInputSource(cloud_object);
	align.setSourceFeatures(object_features);
	align.setInputTarget(cloud_scene_downsampled);
	align.setTargetFeatures(scene_features);
	align.setMaximumIterations(50000);  // Number of RANSAC iterations.
	align.setNumberOfSamples(3);  // Number of points to sample for generating/prerejecting a pose.
	align.setCorrespondenceRandomness(5);  // Number of nearest features to use.
	align.setSimilarityThreshold(0.95f);  // Polygonal edge length similarity threshold.
	align.setMaxCorrespondenceDistance(2.5f * leaf);  // Inlier threshold.
	align.setInlierFraction(0.25f);  // Required inlier fraction for accepting a pose hypothesis.

	const auto start_time(std::chrono::high_resolution_clock::now());
	pcl::PointCloud<pcl::PointNormal>::Ptr object_aligned(new pcl::PointCloud<pcl::PointNormal>);
	align.align(*object_aligned);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Point cloud registered: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	std::cout << "Converged = " << align.hasConverged() << ", score = " << align.getFitnessScore() << std::endl;
	std::cout << "Inliers = " << align.getInliers().size() << " / " << cloud_object->size() << " = " << (float(align.getInliers().size()) / float(cloud_object->size())) std::endl;
	std::cout << "Transformation:\n" << align.getFinalTransformation() << std::endl;
	
	//--------------------
	// Visualize.
	pcl::visualization::PCLVisualizer viewer("Alignment");
	viewer.addPointCloud(cloud_scene_downsampled, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(cloud_scene_downsampled, 0.0, 255.0, 0.0), "cloud_scene_downsampled");
	viewer.addPointCloud(object_aligned, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(object_aligned, 0.0, 0.0, 255.0), "object_aligned");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
		//std::this_thread::sleep_for(100ms);
	}
}

void visualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, const pcl::PointCloud<pcl::PointXYZ>::Ptr result)
{
	pcl::visualization::PCLVisualizer viewer("Point Cloud Registration");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(cloud1, 255, 0, 0);
	viewer.addPointCloud(cloud1, source_color, "Source Cloud");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud2, 0, 255, 0);
	viewer.addPointCloud(cloud2, target_color, "Target Cloud");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> result_color(result, 0, 0, 255);
	viewer.addPointCloud(result, result_color, "Result Cloud");

	viewer.addCoordinateSystem(100.0);
	viewer.initCameraParameters();
	viewer.setBackgroundColor(0, 0, 0);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

void frame_to_frame_registration_test()
{
	const std::string tgt_filepath("/path/to/target.ply");
	const std::string src_filepath("/path/to/source.ply");

	// Load point clouds
	pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud_loaded(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud_loaded(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Loading point clouds..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile<pcl::PointXYZ>(tgt_filepath, *tgt_cloud_loaded) == -1)
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(tgt_filepath, *tgt_cloud_loaded) == -1)
		{
			std::cerr << "File not found, " << tgt_filepath << std::endl;
			return;
		}
		//if (pcl::io::loadPCDFile<pcl::PointXYZ>(src_filepath, *src_cloud_loaded) == -1)
		if (pcl::io::loadPLYFile<pcl::PointXYZ>(src_filepath, *src_cloud_loaded) == -1)
		{
			std::cerr << "File not found, " << src_filepath << std::endl;
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point clouds loaded (#points = " << tgt_cloud_loaded->size() << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

#if 0
	// Downsample
	pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Downsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::VoxelGrid<pcl::PointXYZ> sor;
		//pcl::ApproximateVoxelGrid<pcl::PointXYZ> sor;
		const float leaf_size(5.0f);
		sor.setLeafSize(leaf_size, leaf_size, leaf_size);
		sor.setInputCloud(tgt_cloud_loaded);
		sor.filter(*tgt_cloud);
		sor.setInputCloud(src_cloud_loaded);
		sor.filter(*src_cloud);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Downsampled (#points = " << tgt_cloud->size() << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}
#else
	pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_cloud = tgt_cloud_loaded;
	pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud = src_cloud_loaded;
#endif

	// Estimate normals
	pcl::PointCloud<pcl::Normal>::Ptr tgt_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr src_normals(new pcl::PointCloud<pcl::Normal>);
	{
		// NOTE [caution] >>
		//	It is faster to compute normals for original point clouds than for their downsampled point clouds.
		//	It is slower to use pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>::setSearchSurface().

		std::cout << "Estimating normals..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		const double radius_search(5.0);
		ne.setRadiusSearch(radius_search);
		ne.setInputCloud(tgt_cloud);
		//ne.setSearchSurface(tgt_cloud_loaded);
		ne.compute(*tgt_normals);
		ne.setInputCloud(src_cloud);
		//ne.setSearchSurface(src_cloud_loaded);
		ne.compute(*src_normals);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Normals estimated: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	// ICP
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Registering point clouds (ICP)..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		//pcl::IterativeClosestPointWithNormals<pcl::PointXYZ, pcl::PointXYZ> icp;
		//pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
		//icp.setMaximumIterations(50);
		//icp.setTransformationEpsilon(1.0e-8);
		//icp.setTransformationRotationEpsilon(1.0e-8);
		//icp.setEuclideanFitnessEpsilon(1.0);
		//icp.setMaxCorrespondenceDistance(5.0);  // 5m
		//icp.setUseReciprocalCorrespondences(true);
		icp.setInputSource(tgt_cloud);
		icp.setInputTarget(src_cloud);
		icp.align(*icp_result);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point clouds registered (ICP): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		if (icp.hasConverged())
		{
			std::cout << "ICP has converged: score = " << icp.getFitnessScore() << std::endl;
		}
		else
		{
			std::cout << "ICP did not converge." << std::endl;
		}
	}

	// GICP
	pcl::PointCloud<pcl::PointXYZ>::Ptr gicp_result(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Registering point clouds (GICP)..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
		//pcl::GeneralizedIterativeClosestPoint6D gicp;
		//gicp.setMaximumIterations(50);
		//gicp.setTransformationEpsilon(1.0e-8);
		//gicp.setTransformationRotationEpsilon(1.0e-8);
		//gicp.setEuclideanFitnessEpsilon(1.0);
		//gicp.setMaxCorrespondenceDistance(5.0);  // 5m
		//gicp.setUseReciprocalCorrespondences(true);
		gicp.setInputSource(tgt_cloud);
		gicp.setInputTarget(src_cloud);
		gicp.align(*gicp_result);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point clouds registered (GICP): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		if (gicp.hasConverged())
		{
			std::cout << "GICP has converged: score = " << gicp.getFitnessScore() << std::endl;
		}
		else
		{
			std::cout << "GICP did not converge." << std::endl;
		}
	}

	// Joint ICP
	pcl::PointCloud<pcl::PointXYZ>::Ptr jicp_result(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Registering point clouds (Joint ICP)..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::JointIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> jicp;
		//jicp.setMaximumIterations(50);
		//jicp.setTransformationEpsilon(1.0e-8);
		//jicp.setTransformationRotationEpsilon(1.0e-8);
		//jicp.setEuclideanFitnessEpsilon(1.0);
		//jicp.setMaxCorrespondenceDistance(5.0);  // 5m
		//jicp.setUseReciprocalCorrespondences(true);
		jicp.addInputSource(tgt_cloud);
		jicp.addInputTarget(src_cloud);
		jicp.align(*jicp_result);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point clouds registered (Joint ICP): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		if (jicp.hasConverged())
		{
			std::cout << "Joint ICP has converged: score = " << jicp.getFitnessScore() << std::endl;
		}
		else
		{
			std::cout << "Joint ICP did not converge." << std::endl;
		}
	}

	// NDT
	pcl::PointCloud<pcl::PointXYZ>::Ptr ndt_result(new pcl::PointCloud<pcl::PointXYZ>);
	{
		std::cout << "Registering point clouds (NDT)..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
		//ndt.setMaximumIterations(50);
		//ndt.setTransformationEpsilon(1.0e-8);
		//ndt.setTransformationRotationEpsilon(1.0e-8);
		//ndt.setEuclideanFitnessEpsilon(1.0);
		//ndt.setMaxCorrespondenceDistance(5.0);  // 5m
		ndt.setInputSource(tgt_cloud);
		ndt.setInputTarget(src_cloud);
		ndt.align(*ndt_result);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Point clouds registered (NDT): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		if (ndt.hasConverged())
		{
			std::cout << "NDT has converged: score = " << ndt.getFitnessScore() << std::endl;
		}
		else
		{
			std::cout << "NDT did not converge." << std::endl;
		}
	}

	// Visualize the results
	visualize(tgt_cloud, src_cloud, icp_result);
	visualize(tgt_cloud, src_cloud, gicp_result);
	visualize(tgt_cloud, src_cloud, jicp_result);
	visualize(tgt_cloud, src_cloud, ndt_result);
}

// REF [site] >>
//	https://pointclouds.org/documentation/namespacepcl_1_1registration.html
//	https://github.com/PointCloudLibrary/pcl/blob/master/doc/tutorials/content/sources/registration_api/example1.cpp
//	https://github.com/PointCloudLibrary/pcl/blob/master/doc/tutorials/content/sources/registration_api/example2.cpp
//	https://cpp.hotexamples.com/examples/typenamepcl.registration/CorrespondenceRejectorSampleConsensus/-/cpp-correspondencerejectorsampleconsensus-class-examples.html
void correspondence_estimation_test()
{
	throw std::runtime_error("Not yet implemented");

	/*
	// Estimate correspondences.
#if 1
	pcl::registration::CorrespondenceEstimation<MyFeatureType, MyFeatureType> correspondence_est;
#elif 0
	pcl::registration::CorrespondenceEstimationBackProjection<MyFeatureType, MyFeatureType, pcl::Normal> correspondence_est;
#elif 0
	pcl::registration::CorrespondenceEstimationNormalShooting<MyFeatureType, MyFeatureType, pcl::Normal> correspondence_est;
#elif 0
	pcl::registration::CorrespondenceEstimationOrganizedProjection<MyFeatureType, MyFeatureType> correspondence_est;
#endif

	pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
	correspondence_est.setInputSource(cloud_src);
	correspondence_est.setInputTarget(cloud_tgt);
	//if (correspondence_est.requiresSourceNormals())
	//	correspondence_est.setSourceNormals(normals_src);
	//if (correspondence_est.requiresTargetNormals())
	//	correspondence_est.setTargetNormals(normals_tgt);
	//correspondence_est.setIndicesSource(indices_src);
	//correspondence_est.setIndicesTarget(indices_tgt);
	//correspondence_est.setSearchMethodSource(tree);
	//correspondence_est.setSearchMethodTarget(tree);
	correspondence_est.determineCorrespondences(*correspondences);
	//correspondence_est.determineReciprocalCorrespondences(*correspondences);

	std::cout << "#initial correspondences = " << correspondences->size() << std::endl;  // #correspondences = min(#input features, #target features).

	//-----
	// Eliminate duplicate match indices by leaving only 1-1 correspondences.
#if 0
	pcl::registration::CorrespondenceRejectorDistance correspondence_rejector;
	correspondence_rejector.setMaximumDistance(1);  // 1m.
#elif 0
	pcl::registration::CorrespondenceRejectorFeatures correspondence_rejector;
#elif 0
	pcl::registration::CorrespondenceRejectorMedianDistance correspondence_rejector;
	correspondence_rejector.setMedianFactor(8.79241104);
#elif 1
	pcl::registration::CorrespondenceRejectorOneToOne correspondence_rejector;
#elif 0
	pcl::registration::CorrespondenceRejectionOrganizedBoundary correspondence_rejector;
#elif 0
	pcl::registration::CorrespondenceRejectorPoly correspondence_rejector;
#elif 0
	pcl::registration::CorrespondenceRejectorSurfaceNormal correspondence_rejector;
	correspondence_rejector.setThreshold(std::acos(deg2rad(45.0)));
	correspondence_rejector.initializeDataContainer<pcl::PointXYZ, pcl::PointNormal>();
	correspondence_rejector.setInputCloud<pcl::PointXYZ>(cloud_src);
	correspondence_rejector.setInputTarget<pcl::PointXYZ>(cloud_tgt);
	correspondence_rejector.setInputNormals<pcl::PointXYZ, pcl::PointNormal>(normals_src);
	correspondence_rejector.setTargetNormals<pcl::PointXYZ, pcl::PointNormal>(normals_tgt);
#elif 0
	pcl::registration::CorrespondenceRejectorTrimmed correspondence_rejector;
#elif 0
	pcl::registration::CorrespondenceRejectorVarTrimmed correspondence_rejector;
#endif

	pcl::CorrespondencesPtr correspondences_rejected(new pcl::Correspondences());
	//if (correspondence_rejector.requiresSourcePoints())
	//	correspondence_rejector.setSourcePoints(cloud_src);
	//if (correspondence_rejector.requiresTargetPoints())
	//	correspondence_rejector.setTargetPoints(cloud_tgt);
	//if (correspondence_rejector.requiresSourceNormals())
	//	correspondence_rejector.setSourceNormals(normals_src);
	//if (correspondence_rejector.requiresTargetNormals())
	//	correspondence_rejector.setTargetNormals(normals_tgt);
	correspondence_rejector.setInputCorrespondences(correspondences);
	correspondence_rejector.getCorrespondences(*correspondences_rejected);
	//correspondence_rejector.getRemainingCorrespondences(*correspondences, *correspondences_rejected);
	//const double correspondence_score = correspondence_rejector.getCorrespondenceScore(correspondence);
	//const double correspondence_score_normals = correspondence_rejector.getCorrespondenceScoreFromNormals(getCorrespondenceScoreFromNormals);

	std::cout << "#correspondences rejected = " << correspondences_rejected->size() << std::endl;

	//-----
	// Reject correspondences using RANSAC.
#elif 1
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector_sac;
#elif 0
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointWithScale> rejector_sac;
#elif 0
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZI> rejector_sac;
#elif 0
	pcl::registration::CorrespondenceRejectorSampleConsensus2D<pcl::PointXYZ> rejector_sac;
#endif

	pcl::CorrespondencesPtr correspondences_filtered(new pcl::Correspondences());
	rejector_sac.setInputSource(cloud_src);
	rejector_sac.setInputTarget(cloud_tgt);
	rejector_sac.setInlierThreshold(5.0);  // Distance in m, not the squared distance. {5.0, 10.0, 20.0*, 30.0}.
	rejector_sac.setMaximumIterations(10000);  // {500, 1000*, 2000, 5000}.
	rejector_sac.setRefineModel(false);
	rejector_sac.setSaveInliers(true);
	rejector_sac.setInputCorrespondences(correspondences_rejected);
	rejector_sac.getCorrespondences(*correspondences_filtered);
	//rejector_sac.getRemainingCorrespondences(*correspondences_rejected, *correspondences_filtered);
	//rejector_sac.getRemainingCorrespondences(*correspondences, *correspondences_filtered);
	pcl::Indices inliner_indices;
	rejector_sac.getInliersIndices(inliner_indices);
	*/
}

// REF [site] >> https://pointclouds.org/documentation/namespacepcl_1_1registration.html
void transformation_estimation_test()
{
	throw std::runtime_error("Not yet implemented");

	/*
#if 
	pcl::registration::TransformationEstimation2D<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimation3Point<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationDQ<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationDualQuaternion<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationPointToPlaneWeighted<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 1
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#elif 0
	pcl::registration::TransformationEstimationSVDScale<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
#endif

	Eigen::Matrix4f T(Eigen::Matrix4f::Identity());
	//transformation_estimation.estimateRigidTransformation(cloud_src, cloud_tgt, T);
	//transformation_estimation.estimateRigidTransformation(cloud_src, indices_src, cloud_tgt, T);
	//transformation_estimation.estimateRigidTransformation(cloud_src, indices_src, cloud_tgt, indices_tgt, T);
	transformation_estimation.estimateRigidTransformation(cloud_src, cloud_tgt, correspondences, T);

	//-----
#if 
	pcl::registration::TransformationValidation<pcl::PointXYZ, pcl::PointXYZ> transformation_validation;
#elif 0
	pcl::registration::TransformationValidationEuclidean<pcl::PointXYZ, pcl::PointXYZ> transformation_validation;
	transformation_validation.setSearchMethodTarget(tree);
	transformation_validation.setMaxRange(1.0);
	transformation_validation.setThreshold(1.0);
#endif

	const double score = transformation_validation.validateTransformation(cloud_src, cloud_tgt, transformation_matrix);
	const bool is_valid = transformation_validation.isValid(cloud_src, cloud_tgt, transformation_matrix);
	*/
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/test/registration/test_sac_ia.cpp
void sac_ic_test()
{
	using PointT = pcl::PointXYZ;

	const std::string source_filepath("/path/to/src_sample.pcd");
	const std::string source_filepath("/path/to/tgt_sample.pcd");

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

		std::cout << "Converged = " << sac_ia.hasConverged() << ", score = " << sac_ia.getFitnessScore() << std::endl;
		std::cout << "Transformation:\n" << sac_ia.getFinalTransformation() << std::endl;
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

		std::cout << "Converged = " << sac_ia.hasConverged() << ", score = " << sac_ia.getFitnessScore() << std::endl;
		std::cout << "Transformation:\n" << sac_ia.getFinalTransformation() << std::endl;
	}
}

// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/test/registration/test_fpcs_ia.cpp
void fpcs_test()
{
	using PointT = pcl::PointXYZ;

	const std::string source_filepath("/path/to/src_sample.pcd");
	const std::string source_filepath("/path/to/tgt_sample.pcd");

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

		std::cout << "Converged = " << fpcs_ia.hasConverged() << ", score = " << fpcs_ia.getFitnessScore() << std::endl;
		std::cout << "Transformation:\n" << fpcs_ia.getFinalTransformation() << std::endl;
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

	const std::string source_filepath("/path/to/src_sample.pcd");
	const std::string source_filepath("/path/to/tgt_sample.pcd");

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

		std::cout << "Converged = " << kfpcs_ia.hasConverged() << ", score = " << kfpcs_ia.getFitnessScore() << std::endl;
		std::cout << "Transformation:\n" << kfpcs_ia.getFinalTransformation() << std::endl;
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
	//local::icp_registration_tutorial();
	//local::icp_registration_example();
	//local::ndt_example();  // Not yet tested.
	//local::sample_consensus_prerejective_example();  // Not yet tested.

	local::frame_to_frame_registration_test();  // ICP, GICP, JICP, NDT.

	//local::correspondence_estimation_test();  // Not yet implemented.
	//local::transformation_estimation_test();  // Not yet implemented.

	// Initial alignment.
	//local::sac_ic_test();  // NOTE [info] >> Slow. (~20 secs)
	//local::fpcs_test();  // NOTE [info] >> The alignment results are unstable.
	//local::kfpcs_test();  // NOTE [info] >> Too slow.
}

} // namespace my_pcl
