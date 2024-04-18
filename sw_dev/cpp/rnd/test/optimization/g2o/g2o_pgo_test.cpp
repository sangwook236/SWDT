//#include "stdafx.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <iostream>
#include <Eigen/Core>
#include <pcl/common/common.h>  
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#if defined(G2O_HAVE_CHOLMOD)
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#else
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#endif
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/parameter_camera.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d/edge_se3.h>


namespace {
namespace local {

#if defined(G2O_HAVE_CHOLMOD)
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif
G2O_USE_OPTIMIZATION_LIBRARY(dense);

// REF [site] >> match_point_cloud_features_test() in test/pcl_test/feature_matching.cpp.
bool load_filenames(const std::string &dir_path, std::list<std::string> &input_filepaths)
{
	try
	{
		for (auto const& entry : std::filesystem::directory_iterator{dir_path}) 
			//if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".pcd")
			if (std::filesystem::is_regular_file(entry) && entry.path().extension().compare(".pcd") == 0)
				input_filepaths.push_back(entry.path());
	}
	catch (const std::filesystem::filesystem_error& ex)
	{
		std::cerr << ex.what() << std::endl;
		return false;
	}

	// Sort by filename.
	input_filepaths.sort([](const std::filesystem::path &lhs, const std::filesystem::path &rhs)
	{
		try
		{
			const std::string &lhs_s = lhs.stem().string(), &rhs_s = rhs.stem().string();
			const size_t lhs_pos = lhs_s.find("_"), rhs_pos = rhs_s.find("_");
			const int &lhs_num = std::stoi(lhs_s.substr(lhs_pos + 1)), &rhs_num = std::stoi(rhs_s.substr(rhs_pos + 1));
			return lhs_num < rhs_num;
		}
		catch (const std::exception &ex)
		{
			std::cerr << "Invalid filenames: " << lhs << ", " << rhs << ": " << ex.what() << std::endl;
			return lhs.string() < rhs.string();
		}
	});

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

void pgo_test()
{
	using point_type = pcl::PointXYZ;
	using points_type = pcl::PointCloud<point_type>;
	using points_ptr_type = points_type::Ptr;
	using normal_type = pcl::Normal;
	using normals_type = pcl::PointCloud<normal_type>;
	using normals_ptr_type = normals_type::Ptr;
	using camera_pose_type = Eigen::Affine3f;
	using point_set_type = std::tuple<points_ptr_type, points_ptr_type, normals_ptr_type, camera_pose_type>;

	const bool visualize = false;

	//const std::string dir_path("../../20230317/1_full_arch_pcd");
	//const std::string dir_path("../../20230317/2_occlusal_surface_pcd");
	//const std::string dir_path("../../20230317/3_labial_surface_pcd");
	//const std::string dir_path("../../20230317/4_lingual_surface_pcd");
	const std::string dir_path("../../20230317/5_molar_tilt_pcd");
	//const std::string dir_path("../../20230317/6_canine_tilt_pcd");
	//const std::string dir_path("../../20230317/7_incisor_tilt_pcd");

	// Load point clouds.
	std::list<std::string> input_filepaths;
	{
		std::cout << "Loading input filenames..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		if (!local::load_filenames(dir_path, input_filepaths))
		{
			std::cout << "Failed to load input files from " << dir_path << std::endl;
			return;
		}
		std::cout << "Input filenames loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	}

	// Construct point sets.
	std::vector<point_set_type> point_sets;
	point_sets.reserve(input_filepaths.size());
	{
		std::cout << "Constructing point sets (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const float leaf_size = 20.0f;
		const double search_radius = 20.0;

		//for (size_t fidx = 0; fidx < input_filepaths.size(); ++fidx)
		for (size_t fidx = 0; fidx < input_filepaths.size(); fidx += 50)
		{
			auto it = input_filepaths.begin();
			std::advance(it, fidx);

			auto cloud(std::make_shared<points_type>());
			{
				// Load files.
				std::cout << "Loading a point cloud from " << *it << "..." << std::endl;
				const auto start_time(std::chrono::high_resolution_clock::now());
				if (-1 == pcl::io::loadPCDFile<point_type>(*it, *cloud))
				{
					std::cerr << "File not found, " << *it << std::endl;
					continue;
				}
				std::cout << "Point cloud loaded: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
				std::cout << cloud->size() << " data points (" << pcl::getFieldsList(*cloud) << ") loaded." << std::endl;
			}

			// Construct a point set.
			std::cout << "Constructing a point set..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			auto cloud_downsampled(std::make_shared<points_type>());
			auto cloud_normal(std::make_shared<normals_type>());
			{
				auto tree(std::make_shared<pcl::search::KdTree<point_type> >());

				// Downsample.
				pcl::VoxelGrid<point_type> downsampler;
				downsampler.setLeafSize(leaf_size, leaf_size, leaf_size);  // {5.0, 10.0, 20.0*}.
				downsampler.setInputCloud(cloud);
				downsampler.filter(*cloud_downsampled);

				// Normals.
				//pcl::NormalEstimation<point_type, normal_type> normal_estimation;
				pcl::NormalEstimationOMP<point_type, normal_type> normal_estimation;
				normal_estimation.setSearchMethod(tree);
				normal_estimation.setRadiusSearch(search_radius);  // {20.0*, 40.0}.
				normal_estimation.setInputCloud(cloud);
				//normal_estimation.setInputCloud(cloud_downsampled);
				//normal_estimation.setSearchSurface(cloud);
				normal_estimation.compute(*cloud_normal);
			}
			std::cout << "Point set constructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

			// Approximate initial camera poses.
			const float cx = 150.0f, cy = 225.0f;
			std::map<float, size_t> dists;
			size_t idx = 0;
			for (const auto &pt: cloud->points)
				dists.insert(std::make_pair(std::pow(pt.x - cx, 2) + std::pow(pt.y - cy, 2), idx++));
			const auto &nearest_pt = cloud->points[dists.begin()->second];
			// FIXME [check] >> Initial camera pose. Camera offset in the z direction.
			const Eigen::Affine3f Tcam(Eigen::Translation3f(cx, cy, nearest_pt.z * 2.0f) * Eigen::AngleAxisf(M_PIf, Eigen::Vector3f(1.0f, 0.0f, 0.0f)));

			point_sets.push_back(std::make_tuple(cloud, cloud_downsampled, cloud_normal, Tcam));
		}
		std::cout << point_sets.size() << " point sets constructed (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
	}
	assert(point_sets.size() > 1);

	// Register point sets.
	// {target ID, source ID, converged, fitness score, transformation, inlier indices, correspondences}.
	//std::vector<std::tuple<size_t, size_t, bool, double, Eigen::Matrix4f, pcl::IndicesPtr, pcl::CorrespondencesConstPtr> > icp_results;
	std::vector<std::tuple<size_t, size_t, bool, double, Eigen::Matrix4f, pcl::CorrespondencesConstPtr> > icp_results;
	icp_results.reserve(point_sets.size() - 1);
	std::map<size_t, Eigen::Matrix4f> absolute_transforms;
	absolute_transforms[0] = Eigen::Matrix4f::Identity();
	{
		std::cout << "Registering point sets (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const double max_correspondence_distance = 100.0;
		const int max_iterations = 50;
		const double transformation_epsilon = 0.01;
		const double euclidean_fitness_epsilon = 0.01;

		// ICP.
		pcl::IterativeClosestPoint<point_type, point_type> icp;

		for (size_t sidx = 1; sidx < point_sets.size(); ++sidx)
		{
			const auto tidx = sidx - 1;
			std::cout << "Point set: target ID = " << tidx << ", source ID = " << sidx << std::endl;

			const auto &tgt_point_set = point_sets[tidx];
			const auto &src_point_set = point_sets[sidx];

#if 0
			// This part is not yet tested.

			auto correspodence_est(std::make_shared<pcl::registration::CorrespondenceEstimationBackProjection<point_type, point_type> >());
			correspodence_est->setVoxelRepresentationTarget(dt);
			correspodence_est->setInputSource(std::get<1>(src_point_set));
			correspodence_est->setInputTarget(std::get<1>(tgt_point_set));
			//correspodence_est->setMaxCorrespondenceDistance(max_corr_distance);
			icp.setCorrespondenceEstimation(correspodence_est);

			auto correspodence_rej(std::make_shared<pcl::registration::CorrespondenceRejectorSampleConsensus<point_type> >());
			correspodence_rej->setInputSource(std::get<1>(src_point_set));
			correspodence_rej->setInputTarget(std::get<1>(tgt_point_set));
			//correspodence_rej->setMaximumIterations(max_iterations);
			//correspodence_rej->setInlierThreshold(inlier_threshold);
			icp.addCorrespondenceRejector(correspodence_rej);

			auto transformation_est(std::make_shared<pcl::registration::TransformationEstimationSVD<point_type, point_type> >());
			icp.setTransformationEstimation(transformation_est);
#endif

			// Align downsampled point clouds.
			points_type src_points_transformed;
			{
				icp.setMaxCorrespondenceDistance(max_correspondence_distance);  // 2.5 * downsample leaf size.
				icp.setMaximumIterations(max_iterations);  // {10, 100, 1000}.
				icp.setTransformationEpsilon(transformation_epsilon * 0.1);
				icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon * 0.1);

				icp.setInputSource(std::get<1>(src_point_set));
				icp.setInputTarget(std::get<1>(tgt_point_set));

				//icp.align(src_points_transformed, Tinit);
				icp.align(src_points_transformed);
			}

			// Align raw point clouds.
			{
				const auto &T = icp.getFinalTransformation();

				//icp.setMaxCorrespondenceDistance(max_correspondence_distance);  // 2.5 * downsample leaf size.
				//icp.setMaximumIterations(max_iterations);  // {10, 100, 1000}.
				icp.setTransformationEpsilon(transformation_epsilon);
				icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);

				icp.setInputSource(std::get<0>(src_point_set));
				icp.setInputTarget(std::get<0>(tgt_point_set));

				icp.align(src_points_transformed, T);
			}

			if (!icp.hasConverged())
				std::cerr << "\tNot converged." << std::endl;

			icp_results.push_back(std::make_tuple(tidx, sidx, icp.hasConverged(), icp.getFitnessScore(), icp.getFinalTransformation(), icp.correspondences_));

			// Compute absolute transformations.
			if (icp.hasConverged())
			{
				if (absolute_transforms.find(tidx) != absolute_transforms.end())
					absolute_transforms[sidx] = absolute_transforms[tidx] * icp.getFinalTransformation();
				else
				{
					// TODO [improve] >>
					absolute_transforms[tidx] = Eigen::Matrix4f::Identity();
					absolute_transforms[sidx] = icp.getFinalTransformation();
				}
			}
			else
			{
				// TODO [improve] >>
				if (absolute_transforms.find(tidx) == absolute_transforms.end())
					absolute_transforms[tidx] = Eigen::Matrix4f::Identity();
				absolute_transforms[sidx] = Eigen::Matrix4f::Identity();
			}
		}

		std::cout << "Point sets registered (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
	}
	assert(point_sets.size() == absolute_transforms.size());

	// Visualize.
	if (visualize)
	{
		pcl::visualization::PCLVisualizer viewer("ICP Viewer before PGO");
		viewer.addCoordinateSystem(100.0);
		viewer.setBackgroundColor(0, 0, 0);
		viewer.initCameraParameters();
		viewer.setCameraPosition(
			500.0, 1000.0, 1000.0,  // The coordinates of the camera location.
			0.0, 0.0, 0.0,  // The components of the view point of the camera.
			0.0, 0.0, 1.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 6.0);  // [rad].
		//viewer.setCameraClipDistances(0.05, 50);

		for (const auto &trans_pair: absolute_transforms)
		{
			const auto &pcidx = trans_pair.first;
			const auto &T = trans_pair.second;
			const auto &point_set = point_sets[pcidx];

			auto points_viz(std::make_shared<points_type>());
			pcl::transformPointCloud(*std::get<0>(point_set), *points_viz, T);  // Raw point cloud.
			//pcl::transformPointCloud(*std::get<1>(point_set), *points_viz, T);  // Downsampled point cloud.

			const std::string cloud_id("Point Cloud #" + std::to_string(pcidx)), normal_id("Point Cloud Normal #" + std::to_string(pcidx)), camera_id("Camera #" + std::to_string(pcidx));
			viewer.addPointCloud<point_type>(points_viz, cloud_id);
			//viewer.addPointCloudNormals<point_type, normal_type>(std::get<0>(point_set), std::get<2>(point_set), 1, 5.0f, normal_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, cloud_id);
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, cloud_id);
			const auto &Tcaminit = std::get<3>(point_set);
			viewer.addCoordinateSystem(50.0, Eigen::Affine3f(T) * Tcaminit, camera_id);
		}

		//viewer.setRepresentationToSurfaceForAllActors();
		viewer.resetCamera();

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			//std::this_thread::sleep_for(50ms);
		}
	}

	// Pose graph optimization (PGO).
	// Refer to slam2d_tutorial(), slam3d_se3_pointxyz_test(), & slam3d_se3_test().
	{
		std::cout << "Performing PGO (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const double PIXEL_NOISE = 1.0;
		const double OUTLIER_RATIO = 0.0;
		const bool ROBUST_KERNEL = false;
		const bool STRUCTURE_ONLY = false;
		const bool DENSE = false;

		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(false);
#if 1
		std::string solverName = "lm_fix6_3";
		if (DENSE)
		{
			solverName = "lm_dense6_3";
		}
		else
		{
#if defined(G2O_HAVE_CHOLMOD)
			solverName = "lm_fix6_3_cholmod";
#else
			solverName = "lm_fix6_3";
#endif
		}
		g2o::OptimizationAlgorithmProperty solverProperty;
		optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty));
#else
		std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
		if (DENSE)
		{
			linearSolver= g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> >();
			std::cout << "Using DENSE" << std::endl;
		}
		else
		{
			linearSolver= g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> >();
			std::cout << "Using CHOLMOD" << std::endl;
		}
		auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
		optimizer.setAlgorithm(solver);
#endif

		// Add the parameter representing the sensor offset.
		// FIXME [check] >> Camera parameters.
		const double focal_length = 1000.0;
		const Eigen::Vector2d principal_point(150.0, 225.0);  // For 300x450 images.

#if 0
		auto camera_param = new g2o::CameraParameters(focal_length, principal_point, 0.0);
#else
		auto camera_param = new g2o::ParameterCamera();
		camera_param->setKcam(focal_length, focal_length, principal_point[0], principal_point[1]);
#endif
		camera_param->setId(0);

		if (!optimizer.addParameter(camera_param))
		{
			assert(false);
		}

		// Add the camera poses (vertices).
		std::cout << "Optimization: add camera poses..." << std::endl;
		for (const auto &trans_pair: absolute_transforms)
		{
			const auto &pcidx = trans_pair.first;
			const auto &T = trans_pair.second;
			const auto &point_set = point_sets[pcidx];
			const auto &Tcaminit = std::get<3>(point_set);
			const auto Tcam(Eigen::Affine3f(T) * Tcaminit);

			auto camera = new g2o::VertexSE3;
			camera->setId(int(pcidx));
			camera->setEstimate(g2o::SE3Quat(Tcam.rotation().cast<double>(), Tcam.translation().cast<double>()));

			optimizer.addVertex(camera);
		}
		std::cout << "Done." << std::endl;

		// Add the landmark vertices.
		std::cout << "Optimization: add landmark vertices..." << std::endl;
		std::vector<size_t> vertex_id_offsets;
		vertex_id_offsets.reserve(point_sets.size());
		size_t start_vid = absolute_transforms.size();
		for (const auto &point_set: point_sets)
		{
			const auto &cloud = std::get<0>(point_set);  // Raw point cloud.
			//const auto &cloud = std::get<1>(point_set);  // Downsampled point cloud.

			vertex_id_offsets.push_back(start_vid);

			size_t id = start_vid;
			for (const auto &pt: cloud->points)
			{
				auto landmark = new g2o::VertexPointXYZ;
				landmark->setId(int(id++));
				landmark->setEstimate(g2o::Vector3(pt.x, pt.y, pt.z));

				optimizer.addVertex(landmark);
			}
			start_vid += cloud->size();
		}
		std::cout << "Done." << std::endl;

		// Add the odometry constraints.
		std::cout << "Optimization: add odometry measurements..." << std::endl;
		for (const auto &res: icp_results)
		{
			const auto &tidx = std::get<0>(res);
			const auto &sidx = std::get<1>(res);
			//const auto &is_converged = std::get<2>(res);
			//const auto &fitness_score = std::get<3>(res);
			const auto &Trel = std::get<4>(res);
			//const auto &correspondences = std::get<5>(res);

			auto odometry = new g2o::EdgeSE3;
			odometry->vertices()[0] = optimizer.vertex(int(tidx));
			odometry->vertices()[1] = optimizer.vertex(int(sidx));
			// FIXME [check] >> Relative transformation.
			odometry->setMeasurement(g2o::Isometry3(Trel.cast<double>()));
			// FIXME [check] >> Information matrix.
			odometry->setInformation(Eigen::Matrix<double, 6, 6>::Identity());

			optimizer.addEdge(odometry);
		}
		std::cout << "Done." << std::endl;

		// Add the landmark constraints.
		std::cout << "Optimization: add landmark observations..." << std::endl;
		for (const auto &res: icp_results)
		{
			const auto &tidx = std::get<0>(res);
			const auto &sidx = std::get<1>(res);
			//const auto &is_converged = std::get<2>(res);
			//const auto &fitness_score = std::get<3>(res);
			//const auto &Trel = std::get<4>(res);
			const auto &correspondences = std::get<5>(res);

			const auto &tcloud = std::get<0>(point_sets[tidx]);  // Raw point cloud.
			const auto &scloud = std::get<0>(point_sets[sidx]);  // Raw point cloud.

			const auto &tid_offset = vertex_id_offsets[tidx];
			const auto &sid_offset = vertex_id_offsets[sidx];
			for (const auto &corr: *correspondences)
			{
				const auto &tpt = tcloud->points[corr.index_match];
				const auto &spt = scloud->points[corr.index_query];

				auto landmarkObservation = new g2o::EdgeSE3PointXYZ;
				landmarkObservation->vertices()[0] = optimizer.vertex(int(tid_offset + corr.index_match));
				landmarkObservation->vertices()[1] = optimizer.vertex(int(sid_offset + corr.index_query));
				// FIXME [check] >> Measurement == distance vector.
				landmarkObservation->setMeasurement(g2o::Vector3(spt.x, spt.y, spt.z) - g2o::Vector3(tpt.x, tpt.y, tpt.z));
				// FIXME [check] >> Information matrix.
				landmarkObservation->setInformation(Eigen::Matrix3d::Identity());
				landmarkObservation->setParameterId(0, camera_param->id());

				optimizer.addEdge(landmarkObservation);
			}
		}
		std::cout << "Done." << std::endl;

		// Optimization.
		{
			// Dump initial state to the disk.
			//optimizer.save("../pgo_before.g2o");

			// Prepare and run the optimization.
			// Fix the first camera pose to account for gauge freedom.
			auto firstCameraPose = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(0));
			firstCameraPose->setFixed(true);

			std::cout << "Optimizing..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());
			optimizer.initializeOptimization();
			optimizer.setVerbose(true);
			optimizer.optimize(10);
			std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;

			//optimizer.save("../pgo_after.g2o");

			// FIXME [delete] >>
			{
				const auto &T = absolute_transforms[0];
				const auto &point_set = point_sets[0];
				const auto &cloud = std::get<0>(point_set);
				const auto &Tcaminit = std::get<3>(point_set);
				const auto Tcam(Eigen::Affine3f(T) * Tcaminit);

				const auto &camera_estimate = static_cast<g2o::VertexSE3 *>(optimizer.vertex(0))->estimate();  // Camera #1.
				const auto &landmark_estimate = static_cast<g2o::VertexPointXYZ *>(optimizer.vertex(int(point_sets.size() + 0)))->estimate();  // Landmark #1.

				std::cout << "*****" << std::endl;
				std::cout << Tcam.translation() << std::endl;
				std::cout << Tcam.rotation() << std::endl;
				std::cout << cloud->points[0] << std::endl;
				std::cout << camera_estimate.translation() << std::endl;
				std::cout << camera_estimate.rotation() << std::endl;
				std::cout << landmark_estimate << std::endl;
			}

			// Free the graph memory.
			optimizer.clear();
		}

		std::cout << "PGO performed (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
	}

	// Visualize.
	if (visualize)
	{
		pcl::visualization::PCLVisualizer viewer("ICP Viewer after PGO");
		viewer.addCoordinateSystem(100.0);
		viewer.setBackgroundColor(0, 0, 0);
		viewer.initCameraParameters();
		viewer.setCameraPosition(
			500.0, 1000.0, 1000.0,  // The coordinates of the camera location.
			0.0, 0.0, 0.0,  // The components of the view point of the camera.
			0.0, 0.0, 1.0  // The component of the view up direction of the camera.
		);
		viewer.setCameraFieldOfView(M_PI / 6.0);  // [rad].
		//viewer.setCameraClipDistances(0.05, 50);

		for (const auto &trans_pair: absolute_transforms)
		{
			const auto &pcidx = trans_pair.first;
			const auto &T = trans_pair.second;
			const auto &point_set = point_sets[pcidx];

			auto points_viz(std::make_shared<points_type>());
			pcl::transformPointCloud(*std::get<0>(point_set), *points_viz, T);  // Raw point cloud.
			//pcl::transformPointCloud(*std::get<1>(point_set), *points_viz, T);  // Downsampled point cloud.

			const std::string cloud_id("Point Cloud #" + std::to_string(pcidx)), normal_id("Point Cloud Normal #" + std::to_string(pcidx)), camera_id("Camera #" + std::to_string(pcidx));
			viewer.addPointCloud<point_type>(points_viz, cloud_id);
			//viewer.addPointCloudNormals<point_type, normal_type>(std::get<0>(point_set), std::get<2>(point_set), 1, 5.0f, normal_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, cloud_id);
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, cloud_id);
			const auto &Tcaminit = std::get<3>(point_set);
			viewer.addCoordinateSystem(50.0, Eigen::Affine3f(T) * Tcaminit, camera_id);
		}

		//viewer.setRepresentationToSurfaceForAllActors();
		viewer.resetCamera();

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			//std::this_thread::sleep_for(50ms);
		}
	}
}
