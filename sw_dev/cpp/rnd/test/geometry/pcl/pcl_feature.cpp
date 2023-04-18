#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
//#include <pcl/gpu/features/features.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std::literals::chrono_literals;


namespace {
namespace local {

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/how_features_work.html
void normal_estimation_tutorial()
{
	const std::string filename("./sample.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
	{
		const std::string err("File not found " + filename + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

#if 1
	// Create the normal estimation class, and pass the input dataset to it.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);

	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	// Output datasets.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

	// Use all neighbors in a sphere of radius 3cm.
	ne.setRadiusSearch(0.03);

	// Compute the features.
	ne.compute(*cloud_normals);

	// cloud_normals->size() should have the same size as the input cloud->size().
#elif 0
	// Create the normal estimation class, and pass the input dataset to it.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);

	// Create a set of indices to be used. For simplicity, we're going to be using the first 10% of the points in cloud.
	std::vector<int> indices(std::floor(cloud->size() / 10));
	for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;

	// Pass the indices.
	pcl::shared_ptr<std::vector<int> > indicesptr(new std::vector<int>(indices));
	ne.setIndices(indicesptr);

	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	// Output datasets.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

	// Use all neighbors in a sphere of radius 3cm.
	ne.setRadiusSearch(0.03);

	// Compute the features.
	ne.compute(*cloud_normals);

	// cloud_normals->size() should have the same size as the input indicesptr->size().
#else
	const std::string filename_downsampled("./sample_downsampled.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename_downsampled, *cloud_downsampled) == -1)
	{
		const std::string err("File not found " + filename_downsampled + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	// Create the normal estimation class, and pass the input dataset to it.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_downsampled);

	// Pass the original data (before downsampling) as the search surface.
	ne.setSearchSurface(cloud);

	// Create an empty kdtree representation, and pass it to the normal estimation object.
	// Its content will be filled inside the object, based on the given surface dataset.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	// Output datasets.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

	// Use all neighbors in a sphere of radius 3cm.
	ne.setRadiusSearch(0.03);

	// Compute the features.
	ne.compute(*cloud_normals);

	// cloud_normals->size() should have the same size as the input cloud_downsampled->size().
#endif
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation_using_integral_images.html
void normal_estimation_using_integral_images_tutorial()
{
	const std::string filename("./table_scene_mug_stereo_textured.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
	{
		const std::string err("File not found " + filename + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	// Estimate normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normals);

	// Visualize normals.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

// REF [site] >> https://github.com/otherlab/pcl/blob/master/examples/features/example_principal_curvatures_estimation.cpp
void principal_curvature_estimation_example()
{
	const std::string filepath("../table_scene_lms400.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *cloud) == -1)
	{
		const std::string err("File not found " + filepath + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	std::cout << "Loaded " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

#if 1
	// Create the filtering object.
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	//sor.setMinimumPointsNumberPerVoxel(0);
	sor.filter(*cloud);

	std::cerr << "Filtered " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;
#endif

	// Compute the normals.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::Normal>);
	{
		const auto start_time(std::chrono::high_resolution_clock::now());

		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
		normal_estimation.setInputCloud(cloud);
		normal_estimation.setSearchMethod(tree);
		normal_estimation.setRadiusSearch(0.03);
		normal_estimation.compute(*cloud_with_normals);

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Normals computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;
	}

	// Compute the principal curvatures.
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	{
		const auto start_time(std::chrono::high_resolution_clock::now());

		// Setup the principal curvatures computation.
		pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
		// Provide the original point cloud (without normals).
		principal_curvatures_estimation.setInputCloud(cloud);
		// Provide the point cloud with normals.
		principal_curvatures_estimation.setInputNormals(cloud_with_normals);
		// Use the same KdTree from the normal estimation.
		principal_curvatures_estimation.setSearchMethod(tree);
		principal_curvatures_estimation.setRadiusSearch(0.01);
		//principal_curvatures_estimation.setKSearch(5);
		// Actually compute the principal curvatures.
		principal_curvatures_estimation.compute(*principal_curvatures);

		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Principal curvatures computed: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

		std::cout << "#principal curvatures = " << principal_curvatures->size() << std::endl;  // #principal curvatures = #input points.
	}

	// Display and retrieve the shape context descriptor vector for the 0th point.
	const auto point_id = 0;
	const auto &pt = (*cloud)[point_id];
	//const auto &pt = cloud->at(point_id);
	std::cout << "Principal curvatures at point #" << point_id << " (" << pt.x << ", " << pt.y << ", " << pt.z << ")." << std::endl;
	const pcl::PrincipalCurvatures &descriptor = principal_curvatures->points[point_id];
	//const pcl::PrincipalCurvatures &descriptor = principal_curvatures->at(point_id);
	std::cout << "\tDescriptor: " << descriptor << std::endl;
	std::cout << "\tThe max and min eigenvalues of curvature: pc1 = " << descriptor.pc1 << ", pc2 = " << descriptor.pc2 << std::endl;
	std::cout << "\tPrincipal direction (eigenvector of the max eigenvalue): (" << descriptor.principal_curvature_x << ", " << descriptor.principal_curvature_y << ", " << descriptor.principal_curvature_z << ")." << std::endl;

#if 1
	// Filter the valid principal curvatures.
	pcl::Indices valid_point_indices;
	//std::ofstream stream("../principal_curvatures.csv");
	int point_idx = 0;
	for (const auto &curvature: principal_curvatures->points)
	{
		//curvature.principal_curvature_x;  // The principal curvature X direction.
		//curvature.principal_curvature_y;  // The principal curvature Y direction.
		//curvature.principal_curvature_z;  // The principal curvature Z direction.
		//curvature.pc1;  // The max eigenvalue of curvature.
		//curvature.pc2;  // The min eigenvalue of curvature.

		//if (!std::isnan(curvature.pc1) && !std::isnan(curvature.pc2))
		if (!std::isnan(curvature.pc1) && !std::isnan(curvature.pc2) && curvature.pc1 > 0.00005f)
		//if (!std::isnan(curvature.pc1) && !std::isnan(curvature.pc2) && curvature.pc1 > 0.00005f && curvature.pc2 > 0.000005f)
		//if (!std::isnan(curvature.pc1) && !std::isnan(curvature.pc2) && curvature.pc2 > 0.0f && curvature.pc1 / curvature.pc2 > 100.0f)  // Not good.
		{
			valid_point_indices.push_back(point_idx);

			//stream << curvature.pc1 << ", " << curvature.pc2 << std::endl;
		}

		++point_idx;
	}
	//stream.close();

	std::cout << "#valid principal curvatures = " << valid_point_indices.size() << std::endl;

	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(*cloud, valid_point_indices));
	cloud_with_normals.reset(new pcl::PointCloud<pcl::Normal>(*cloud_with_normals, valid_point_indices));
	principal_curvatures.reset(new pcl::PointCloud<pcl::PrincipalCurvatures>(*principal_curvatures, valid_point_indices));

	std::cout << "Valid points: " << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;
	std::cout << "Valid normals: " << cloud_with_normals->size() << " data points (" << pcl::getFieldsList(*cloud_with_normals) << ")." << std::endl;
	std::cout << "Valid principal curvatures: " << principal_curvatures->size() << " data points (" << pcl::getFieldsList(*principal_curvatures) << ")." << std::endl;
#endif

	// Visualize.
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
	//viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_with_normals, 5, 0.01f, "cloud_normals");
	viewer.addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, cloud_with_normals, principal_curvatures, 5, 0.01f, "cloud_curvatures");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "cloud");
	//viewer.setRepresentationToSurfaceForAllActors();
	//viewer.addCoordinateSystem(0.3);
	viewer.initCameraParameters();
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

void principal_curvature_estimation_gpu_test()
{
	throw std::runtime_error("Not yet implemented");

	//pcl::gpu::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/pfh_estimation.html
void pfh_estimation_tutorial()
{
	const std::string filename("./sample.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
	{
		const std::string err("File not found " + filename + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	// REF [function] >> normal_estimation_tutorial().
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);  // Create an empty kdtree representation.
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*normals);
	}

	//--------------------
	// Create the PFH estimation class, and pass the input dataset+normals to it.
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
	pfh.setInputCloud(cloud);
	pfh.setInputNormals(normals);
	//pfh.setInputNormals(cloud);  // Alternatively, if cloud is of type PointNormal.

	// Create an empty kdtree representation, and pass it to the PFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	//pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>());   // Older call for PCL 1.5.
	pfh.setSearchMethod(tree);

	// Use all neighbors in a sphere of radius 5cm.
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	pfh.setRadiusSearch(0.05);

	// Output datasets.
	pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs(new pcl::PointCloud<pcl::PFHSignature125>());

	// Compute the features.
	std::cout << "Describing PFH..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	pfh.compute(*pfhs);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "PFH described: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	// pfhs->size() should have the same size as the input cloud->size().
}

// REF [site] >> https://pcl.readthedocs.io/projects/tutorials/en/latest/fpfh_estimation.html
void fpfh_estimation_tutorial()
{
	const std::string filename("./sample.pcd");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

	// Load the file.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
	{
		const std::string err("File not found " + filename + ".\n");
		PCL_ERROR(err.c_str());
		return;
	}

	// REF [function] >> normal_estimation_tutorial().
	{
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(cloud);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);  // Create an empty kdtree representation.
		ne.setRadiusSearch(0.03);  // Use all neighbors in a sphere of radius 3cm.
		ne.compute(*normals);
	}

	//--------------------
	// Create the FPFH estimation class, and pass the input dataset+normals to it.
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	//pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud);
	fpfh.setInputNormals(normals);
	//fpfh.setInputNormals(cloud);  // Alternatively, if cloud is of type PointNormal.

	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	fpfh.setSearchMethod(tree);

	// Use all neighbors in a sphere of radius 5cm.
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	fpfh.setRadiusSearch(0.05);

	// Output datasets.
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

	// Compute the features.
	std::cout << "Describing FPFH..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	fpfh.compute(*fpfhs);
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "FPFH described: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() / 1000.0f << " secs." << std::endl;

	// fpfhs->size() should have the same size as the input cloud->size().
}

void fpfh_estimation_gpu_test()
{
	throw std::runtime_error("Not yet implemented");

	//pcl::gpu::FPFHEstimation::PointCloud cloud_gpu;
	//pcl::gpu::FPFHEstimation::Normals normals_gpu;
	//pcl::gpu::FPFHEstimation::Indices indices_gpu;
	//pcl::gpu::FPFHEstimation::PointCloud surface_gpu;
	//pcl::gpu::FPFHEstimation fe_gpu;
}

#if 0

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>

double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
			continue;
		// Consider the second neighbor since the first is the point itself.
		const int nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
		res /= n_points;
	return res;
}

// REF [site] >> https://stackoverflow.com/questions/41105987/pcl-feature-matching
void feature_matching()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new PointCloud<PointXYZ>());
	loadPCDFile("./cloudASCII000.pcd", *source_cloud);
	std::cout << "File 1 points: " << source_cloud->size() << std::endl;

	// Compute model resolution.
	double model_resolution = computeCloudResolution(source_cloud);

	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

	iss_detector.setSearchMethod(tree);
	iss_detector.setSalientRadius(10 * model_resolution);
	iss_detector.setNonMaxRadius(8 * model_resolution);
	iss_detector.setThreshold21(0.2);
	iss_detector.setThreshold32(0.2);
	iss_detector.setMinNeighbors(10);
	iss_detector.setNumberOfThreads(10);
	iss_detector.setInputCloud(source_cloud);
	iss_detector.compute(*source_keypoints);
	pcl::PointIndicesConstPtr keypoints_indices = iss_detector.getKeypointsIndices();

	std::cout << "No of ISS points in the result are " << source_keypoints->size() << std::endl;
	pcl::io::savePCDFileASCII("./ISSKeypoints1.pcd", *source_keypoints);

	// Compute the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(source_cloud);
	normalEstimation.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
	normalEstimation.setRadiusSearch(0.2);
	normalEstimation.compute(*source_normals);

#if 1
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	//pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(0.2);
	//fpfh.setKSearch(10);
	fpfh.setInputCloud(source_cloud);
	fpfh.setInputNormals(source_normals);
	fpfh.setIndices(keypoints_indices);
	fpfh.compute(*source_features);
#else
	// SHOT optional descriptor.
	pcl::PointCloud<pcl::SHOT352>::Ptr source_features(new pcl::PointCloud<pcl::SHOT352>());
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	//pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
	//pcl::PointCloud<pcl::SHOT1344>::Ptr source_features(new pcl::PointCloud<pcl::SHOT1344>());
	//pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot;
	//pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot;
	shot.setSearchMethod(tree);
	shot.setRadiusSearch(0.2);
	//shot.setKSearch(10);
	shot.setInputCloud(source_cloud);
	shot.setInputNormals(source_normals);
	shot.setIndices(keypoints_indices);
	shot.compute(*source_features);
#endif

	// Target point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new PointCloud<PointXYZ>());
	loadPCDFile("cloudASCII003.pcd", *target_cloud);
	std::cout << "File 2 points: " << target_cloud->size() << std::endl;

	// Compute model resolution.
	double model_resolution_1 = computeCloudResolution(target_cloud);

	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector_1;
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_keypoints(new pcl::PointCloud<pcl::PointXYZ>());

	iss_detector_1.setSearchMethod(tree);
	iss_detector_1.setSalientRadius(10 * model_resolution_1);
	iss_detector_1.setNonMaxRadius(8 * model_resolution_1);
	iss_detector_1.setThreshold21(0.2);
	iss_detector_1.setThreshold32(0.2);
	iss_detector_1.setMinNeighbors(10);
	iss_detector_1.setNumberOfThreads(10);
	iss_detector_1.setInputCloud(target_cloud);
	iss_detector_1.compute(*target_keypoints);
	pcl::PointIndicesConstPtr keypoints_indices_1 = iss_detector_1.getKeypointsIndices();

	std::cout << "No of ISS points in the result are " << target_keypoints->size() << std::endl;
	pcl::io::savePCDFileASCII("./ISSKeypoints2.pcd", *target_keypoints);

	// Compute the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation_1;
	//pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation_1;
	normalEstimation_1.setInputCloud(target_cloud);
	normalEstimation_1.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
	normalEstimation_1.setRadiusSearch(0.2);
	normalEstimation_1.compute(*target_normals);

#if 1
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh1;
	//pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh1;
	fpfh1.setSearchMethod(tree);
	fpfh1.setRadiusSearch(0.2);
	//fpfh1.setKSearch(10);
	fpfh1.setInputCloud(target_cloud);
	fpfh1.setInputNormals(target_normals);
	fpfh1.setIndices(keypoints_indices_1);
	fpfh1.compute(*target_features);
#else
	// SHOT optional descriptor.
	pcl::PointCloud<pcl::SHOT352>::Ptr target_features(new pcl::PointCloud<pcl::SHOT352>());
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_1;
	//pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_1;
	//pcl::PointCloud<pcl::SHOT1344>::Ptr target_features(new pcl::PointCloud<pcl::SHOT1344>());
	//pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot_1;
	//pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shot_1;
	shot_1.setSearchMethod(tree);
	shot_1.setRadiusSearch(0.2);
	//shot_1.setKSearch(10);
	shot_1.setInputCloud(target_cloud);
	shot_1.setInputNormals(target_normals);
	shot_1.setIndices(keypoints_indices_1);
	shot_1.compute(*target_features);
#endif

	// Estimate correspondences.
	pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
	pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
	est.setInputSource(source_features);
	est.setInputTarget(target_features);
	est.determineCorrespondences(*correspondences);

	// Eliminate duplicate match indices.
	pcl::CorrespondencesPtr correspondences_result_rej_one_to_one(new pcl::Correspondences());
	pcl::registration::CorrespondenceRejectorOneToOne corr_rej_one_to_one;
	corr_rej_one_to_one.setInputCorrespondences(correspondences);
	corr_rej_one_to_one.getCorrespondences(*correspondences_result_rej_one_to_one);

	// Correspondance rejection RANSAC.
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector_sac;
	pcl::CorrespondencesPtr correspondences_filtered(new pcl::Correspondences());
	rejector_sac.setInputSource(source_keypoints);
	rejector_sac.setInputTarget(target_keypoints);
	rejector_sac.setInlierThreshold(2.5);  // Distance in m, not the squared distance.
	rejector_sac.setMaximumIterations(1000000);
	rejector_sac.setRefineModel(false);
	rejector_sac.setInputCorrespondences(correspondences_result_rej_one_to_one);

	rejector_sac.getCorrespondences(*correspondences_filtered);
	correspondences.swap(correspondences_filtered);
	std::cout << correspondences->size() << " vs. " << correspondences_filtered->size() << std::endl;
	transform = rejector_sac.getBestTransformation();  // Transformation Estimation method 1.

	// Transformation Estimation method 2.
	//pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> transformation_estimation;
	//transformation_estimation.estimateRigidTransformation(*source_keypoints, *target_keypoints, *correspondences, transform);
	std::cout << "Estimated transform:" << std::endl << transform << std::endl;

	// Refinement transform source using transformation matrix.

	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final_output(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source_cloud, *transformed_source, transform);
	pcl::io::savePCDFileASCII("./Transformed.pcd", (*transformed_source));

	pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
	//viewer.setBackgroundColor(0, 0, 0);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.resetCamera();
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_source_cloud(transformed_source, 150, 80, 80);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_source_keypoints(source_keypoints, 255, 0, 0);

	viewer.addPointCloud<pcl::PointXYZ>(transformed_source, handler_source_cloud, "source_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source_cloud");
	viewer.addPointCloud<pcl::PointXYZ>(source_keypoints, handler_source_keypoints, "source_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_target_cloud(target_cloud, 80, 150, 80);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_target_keypoints(target_keypoints, 0, 255, 0);

	viewer.addPointCloud<pcl::PointXYZ>(target_cloud, handler_target_cloud, "target_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud");
	viewer.addPointCloud<pcl::PointXYZ>(target_keypoints, handler_target_keypoints, "target_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_keypoints");
	viewer.addCorrespondences<pcl::PointXYZ>(source_keypoints, target_keypoints, *correspondences, 1, "correspondences");

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(transformed_source);
	icp.setInputTarget(target_cloud);
	icp.align(*final_output);
	std::cout << "has converged: " << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;

	pcl::visualization::PCLVisualizer icpViewer("ICP Viewer");
	icpViewer.addPointCloud<pcl::PointXYZ>(final_output, handler_source_cloud, "final_cloud");
	icpViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "source_keypoints");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		std::this_thread::sleep_for(100ms);
	}
	while (!icpViewer.wasStopped())
	{
		icpViewer.spinOnce(100);
		std::this_thread::sleep_for(100ms);
	}
	/*
	// Setup the SHOT features.
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimation;
	//pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimation;
	//pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shotEstimation;
	//pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shotEstimation;
	// Actually compute the spin images.
	pcl::PointCloud<pcl::SHOT352>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT352>);
	// Use the same KdTree from the normal estimation.
	shotEstimation.setSearchMethod(tree);
	//shotEstimation.setRadiusSearch(0.2);
	shotEstimation.setKSearch(10);
	shotEstimation.setInputCloud(model);
	shotEstimation.setInputNormals(normals);
	shotEstimation.setIndices(keypoint_indices);
	shotEstimation.compute(*shotFeatures);
	std::cout << "SHOT output points.size(): " << shotFeatures->size() << std::endl;

	// Display and retrieve the SHOT descriptor for the first point.
	pcl::SHOT352 descriptor = shotFeatures->points[0];
	std::cout << descriptor << std::endl;
	*/
}

#endif

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void feature()
{
	//local::normal_estimation_tutorial();
	//local::normal_estimation_using_integral_images_tutorial();

	local::principal_curvature_estimation_example();
	//local::principal_curvature_estimation_gpu_test();  // Not yet implemented.

	// Point feature histograms (PFH) descriptor.
	//local::pfh_estimation_tutorial();
	// Fast point feature histograms (FPFH) descriptor.
	//local::fpfh_estimation_tutorial();
	//local::fpfh_estimation_gpu_test();  // Not yet implemented.

	//--------------------
	//local::feature_matching();

	// ICCV tutorial.
	// REF [site] >> https://github.com/PointCloudLibrary/pcl/blob/master/apps/src/feature_matching.cpp
}

}  // namespace my_pcl
