//#include "stdafx.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/common/common.h>  
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
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
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#if defined(G2O_HAVE_CHOLMOD)
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#else
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#endif
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/stuff/sampler.h>
#include <g2o/types/slam3d/parameter_camera.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

//#define __USE_SE3_AS_LANDMARKS 1


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

void ba_test()
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

	const std::string dir_path("/path/to/pcd");

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

	const int pg_vertex_start_id = 1;
	std::vector<size_t> pose_graph;  // A simple pose graph as a sequence of point set IDs.
	pose_graph.reserve(input_filepaths.size());

	// Construct point sets.
	std::map<size_t, point_set_type> point_sets;
	{
		std::cout << "Constructing point sets (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const float leaf_size = 20.0f;
		const double search_radius = 20.0;

		auto pg_camera_vertex_id = pg_vertex_start_id;
		for (size_t fidx = 0; fidx < input_filepaths.size(); ++fidx)
		//for (size_t fidx = 0; fidx < input_filepaths.size(); fidx += 10)
		//for (size_t fidx = 0; fidx < input_filepaths.size() && fidx < 50; ++fidx)
		//for (size_t fidx = 0; fidx < input_filepaths.size() && fidx < 20; fidx += 5)
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
			const float cx = 150.0f, cy = 225.0f;  // For 300x450 images.
			const auto &nearest_pt = *std::min_element(cloud->points.begin(), cloud->points.end(), [&cx, &cy](const auto &lhs, const auto &rhs) {
				return std::pow(lhs.x - cx, 2) + std::pow(lhs.y - cy, 2) < std::pow(rhs.x - cx, 2) + std::pow(rhs.y - cy, 2);
			});
			// FIXME [check] >> Initial camera pose. Camera offset in the z direction.
			const camera_pose_type Tcaminit(Eigen::Translation3f(cx, cy, nearest_pt.z + 300.0f) * Eigen::AngleAxisf(M_PIf, Eigen::Vector3f(1.0f, 0.0f, 0.0f)));
			//const camera_pose_type Tcaminit(Eigen::Affine3f::Identity());

			pose_graph.push_back(pg_camera_vertex_id);
			point_sets.insert(std::make_pair(pg_camera_vertex_id, std::make_tuple(cloud, cloud_downsampled, cloud_normal, Tcaminit)));
			++pg_camera_vertex_id;
		}
		std::cout << point_sets.size() << " point sets constructed (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
	}
	assert(point_sets.size() > 1);

	const auto pg_landmark_vertex_start_id = int(pg_vertex_start_id + point_sets.size());
	assert(size_t(pg_landmark_vertex_start_id) >= pg_vertex_start_id + point_sets.size());

	// Create point index to PG landmark ID mappers.
	std::map<size_t, g2o::Vector3> pg_landmarks;
	std::map<size_t, std::vector<size_t> > point_id_to_pg_landmark_id_mappers;
	size_t pg_landmark_vertex_id = pg_landmark_vertex_start_id;
	{
		const auto &cloud = std::get<0>(point_sets[pose_graph[0]]);  // Raw point cloud.
		//const auto &cloud = std::get<1>(point_sets[pose_graph[0]]);  // Downsampled point cloud.

		std::vector<size_t> point_id_to_landmark_id_mapper;  // Point index in point cloud -> landmark ID in pose graph.
		point_id_to_landmark_id_mapper.reserve(cloud->size());
		std::for_each(cloud->points.begin(), cloud->points.end(), [&pg_landmark_vertex_id, &pg_landmarks, &point_id_to_landmark_id_mapper](const auto &pt) {
			pg_landmarks.insert(std::make_pair(pg_landmark_vertex_id, g2o::Vector3(pt.x, pt.y, pt.z)));
			point_id_to_landmark_id_mapper.push_back(pg_landmark_vertex_id);
			++pg_landmark_vertex_id;
		});

		point_id_to_pg_landmark_id_mappers.insert(std::make_pair(pose_graph[0], point_id_to_landmark_id_mapper));
	}
	assert(pg_landmarks.size() == pg_landmark_vertex_id - pg_landmark_vertex_start_id);

	// Register point sets.
	// {target ID, source ID, converged, fitness score, transformation, inlier indices}.
	//std::map<size_t, std::tuple<size_t, bool, double, Eigen::Matrix4f, pcl::IndicesPtr> > icp_results;
	std::map<size_t, std::tuple<size_t, bool, double, Eigen::Matrix4f> > icp_results;
	std::map<size_t, Eigen::Matrix4f> absolute_transforms;
	absolute_transforms[pose_graph[0]] = Eigen::Matrix4f::Identity();
	{
		std::cout << "Registering point sets (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const double max_correspondence_distance = 5.0;
		const int max_iterations = 50;
		const double transformation_epsilon = 0.01;
		const double euclidean_fitness_epsilon = 0.01;

		// ICP.
		pcl::IterativeClosestPoint<point_type, point_type> icp;
		for (size_t nid = 1; nid < pose_graph.size(); ++nid)
		{
			const auto tidx = pose_graph[nid - 1];
			const auto sidx = pose_graph[nid];
			std::cout << "Point set: target ID = " << tidx << ", source ID = " << sidx << std::endl;

			const auto &tgt_point_set = point_sets[tidx];
			const auto &src_point_set = point_sets[sidx];

#if 0
			auto correspodence_est(std::make_shared<pcl::registration::CorrespondenceEstimationBackProjection<point_type, point_type, normal_type> >());
			correspodence_est->setInputSource(std::get<1>(src_point_set));
			correspodence_est->setInputTarget(std::get<1>(tgt_point_set));
			//correspodence_est->setMaxCorrespondenceDistance(max_corr_distance);
			icp.setCorrespondenceEstimation(correspodence_est);

			//auto correspodence_rej(std::make_shared<pcl::registration::CorrespondenceRejectorSampleConsensus<point_type> >());
			auto correspodence_rej(std::make_shared<pcl::registration::CorrespondenceRejectorOneToOne>());
			//correspodence_rej->setInputSource(std::get<1>(src_point_set));
			//correspodence_rej->setInputTarget(std::get<1>(tgt_point_set));
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

			icp_results.insert(std::make_pair(sidx, std::make_tuple(tidx, icp.hasConverged(), icp.getFitnessScore(), icp.getFinalTransformation())));

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

			// Create point index to PG landmark ID mappers.
			{
				const auto &src_cloud = std::get<0>(src_point_set);  // Raw point cloud.
				//const auto &src_cloud = std::get<1>(src_point_set);  // Downsampled point cloud.
				const Eigen::Affine3f T(absolute_transforms[sidx]);
				const auto &tgt_point_id_to_landmark_id_mapper = point_id_to_pg_landmark_id_mappers[tidx];

				std::vector<size_t> src_point_id_to_landmark_id_mapper(src_cloud->size(), size_t(-1));  // Point index in point cloud -> landmark ID in pose graph.
#if 0
				for (const auto &corr: *icp.correspondences_)
					src_point_id_to_landmark_id_mapper[corr.index_query] = tgt_point_id_to_landmark_id_mapper[corr.index_match];
#else
				// Filter one-to-one correspondences (when not using pcl::registration::CorrespondenceRejectorOneToOne).
				const auto &tgt_cloud = std::get<0>(tgt_point_set);  // Raw point cloud.
				//const auto &tgt_cloud = std::get<1>(tgt_point_set);  // Downsampled point cloud.
				std::map<size_t, std::set<size_t> > tgt_match_counts, src_match_counts;
				std::generate_n(std::inserter(tgt_match_counts, tgt_match_counts.end()), tgt_cloud->size(), [idx = 0]() mutable { return std::make_pair(idx++, std::set<size_t>()); });
				std::generate_n(std::inserter(src_match_counts, src_match_counts.end()), src_cloud->size(), [idx = 0]() mutable { return std::make_pair(idx++, std::set<size_t>()); });
				for (const auto &corr: *icp.correspondences_)
				{
					tgt_match_counts[corr.index_match].insert(corr.index_query);
					src_match_counts[corr.index_query].insert(corr.index_match);
				}
				for (const auto &tc: tgt_match_counts)
				{
					if (tc.second.size() == 1)
					{
#if 0
						const auto &smidx = *tc.second.begin();
						if (src_match_counts[smidx].size() == 1 && *src_match_counts[smidx].begin() == tc.first)
							src_point_id_to_landmark_id_mapper[smidx] = tgt_point_id_to_landmark_id_mapper[tc.first];
#else
						src_point_id_to_landmark_id_mapper[*tc.second.begin()] = tgt_point_id_to_landmark_id_mapper[tc.first];
#endif
					}
				}
#endif
				std::cout << "#ICP correspondences = " << icp.correspondences_->size() << ", #one-to-one correspondences filtered = " << std::count_if(src_point_id_to_landmark_id_mapper.begin(), src_point_id_to_landmark_id_mapper.end(), [](const size_t &id) { return size_t(-1) != id; }) << std::endl;

				// Handle unmatched points without correspondences.
				const auto Td(T.cast<double>());
#if 1
				for (size_t idx = 0; idx < src_point_id_to_landmark_id_mapper.size(); ++idx)
					if (size_t(-1) == src_point_id_to_landmark_id_mapper[idx])
					{
						const auto &pt = src_cloud->points[idx];
						//const auto t(Td * Eigen::Vector3d(pt.x, pt.y, pt.z));  // NOTE [error] >> t = (0, 0, 0) in release build. Why?
						const Eigen::Vector3d t(Td * Eigen::Vector3d(pt.x, pt.y, pt.z));
						pg_landmarks.insert(std::make_pair(pg_landmark_vertex_id, t));
						src_point_id_to_landmark_id_mapper[idx] = pg_landmark_vertex_id;
						++pg_landmark_vertex_id;
					}
#else
				for_each(src_point_id_to_landmark_id_mapper.begin(), src_point_id_to_landmark_id_mapper.end(), [&pg_landmarks, &pg_landmark_vertex_id, &src_cloud, &Td](size_t &id) {
					if (size_t(-1) == id)
					{
						const auto &pt = src_cloud->points[?];
						//const auto t(Td * Eigen::Vector3d(pt.x, pt.y, pt.z));  // NOTE [error] >> t = (0, 0, 0) in release build. Why?
						const Eigen::Vector3d t(Td * Eigen::Vector3d(pt.x, pt.y, pt.z));
						pg_landmarks.insert(std::make_pair(pg_landmark_vertex_id, t));
						id = pg_landmark_vertex_id;
						++pg_landmark_vertex_id;
					}
				});
#endif
				assert(pg_landmarks.size() == pg_landmark_vertex_id - pg_landmark_vertex_start_id);

				point_id_to_pg_landmark_id_mappers.insert(std::make_pair(sidx, src_point_id_to_landmark_id_mapper));

#if 0
				{
					// Show ICP correspondence statistics.

					std::map<size_t, size_t> tgt_match_counts, src_match_counts;
					std::generate_n(std::inserter(tgt_match_counts, tgt_match_counts.end()), tgt_cloud->size(), [idx = 0]() mutable { return std::make_pair(idx++, 0); });
					std::generate_n(std::inserter(src_match_counts, src_match_counts.end()), src_cloud->size(), [idx = 0]() mutable { return std::make_pair(idx++, 0); });
					for (const auto &corr: *icp.correspondences_)
					{
						++tgt_match_counts[corr.index_match];
						++src_match_counts[corr.index_query];
					}
					std::map<size_t, size_t> tgt_count_map, src_count_map;
					for (const auto &c: tgt_match_counts)
					{
						if (tgt_count_map.find(c.second) == tgt_count_map.end())
							tgt_count_map[c.second] = 1;
						else
							++tgt_count_map[c.second];
					}
					for (const auto &c: src_match_counts)
					{
						if (src_count_map.find(c.second) == src_count_map.end())
							src_count_map[c.second] = 1;
						else
							++src_count_map[c.second];
					}
					std::cout << "\tTarget (#matches -> #points): ";
					for (const auto &c: tgt_count_map)
						std::cout << c.first << "->" << c.second << ", ";
					std::cout << std::endl;
					std::cout << "\tSource (#matches -> #points): ";
					for (const auto &c: src_count_map)
						std::cout << c.first << "->" << c.second << ", ";
					std::cout << std::endl;
				}
#endif
			}

#if 0
			if (visualize)
			{
				// Visualize ICP correspondences.

				pcl::visualization::PCLVisualizer viewer("ICP Correspondence Viewer before BA");
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

				const auto &tgt_cloud = std::get<0>(tgt_point_set);  // Raw point cloud.
				//const auto &tgt_cloud = std::get<1>(tgt_point_set);  // Downsampled point cloud.
				const auto &src_cloud = std::get<0>(src_point_set);  // Raw point cloud.
				//const auto &src_cloud = std::get<1>(src_point_set);  // Downsampled point cloud.

#if 0
				// Visualize point clouds with absolute transforms.
				const auto &Tt = absolute_transforms[tidx];
				const auto &Ts = absolute_transforms[sidx];

				auto tgt_cloud_viz(std::make_shared<points_type>());
				pcl::transformPointCloud(*tgt_cloud, *tgt_cloud_viz, Tt);
				auto src_cloud_viz(std::make_shared<points_type>());
				pcl::transformPointCloud(*src_cloud, *src_cloud_viz, Ts);
#else
				// Visualize point clouds with relative transforms.
				const auto Tt(Eigen::Matrix4f::Identity());
				const auto &Ts = icp.getFinalTransformation();

				const auto &tgt_cloud_viz = tgt_cloud;
				auto src_cloud_viz(std::make_shared<points_type>());
				pcl::transformPointCloud(*src_cloud, *src_cloud_viz, Ts);
#endif

				viewer.addPointCloud<point_type>(tgt_cloud_viz, "Target Point Cloud");
				//viewer.addPointCloudNormals<point_type, normal_type>(tgt_cloud_viz, tgt_cloud_normal, 1, 5.0f, "Target Point Cloud Normal");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Target Point Cloud");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Target Point Cloud");
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "Target Point Cloud");
				const auto &Ttcaminit = std::get<3>(tgt_point_set);
				viewer.addCoordinateSystem(50.0, Eigen::Affine3f(Tt) * Ttcaminit, "Target Camera");

				viewer.addPointCloud<point_type>(src_cloud_viz, "Source Point Cloud");
				//viewer.addPointCloudNormals<point_type, normal_type>(src_cloud_viz, src_cloud_normal, 1, 5.0f, "Source Point Cloud Normal");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Source Point Cloud");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "Source Point Cloud");
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "Source Point Cloud");
				const auto &Tscaminit = std::get<3>(src_point_set);
				viewer.addCoordinateSystem(25.0, Eigen::Affine3f(Ts) * Tscaminit, "Source Camera");

				viewer.addCorrespondences<point_type>(src_cloud_viz, tgt_cloud_viz, *icp.correspondences_, "Correspondences");

				//viewer.setRepresentationToSurfaceForAllActors();
				viewer.resetCamera();

				while (!viewer.wasStopped())
				{
					viewer.spinOnce();
					//std::this_thread::sleep_for(50ms);
				}
			}
#endif
		}

		std::cout << "Point sets registered (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
		std::cout << "#cameras = " << absolute_transforms.size() << ", #landmarks = " << pg_landmarks.size() << std::endl;
	}
	assert(point_sets.size() == absolute_transforms.size());
	assert(point_sets.size() == point_id_to_pg_landmark_id_mappers.size());
	assert(pg_landmarks.size() == pg_landmark_vertex_id - pg_landmark_vertex_start_id);

	// Visualize.
	if (visualize)
	{
		pcl::visualization::PCLVisualizer viewer("ICP Viewer before BA");
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

		for (const auto &nid: pose_graph)
		{
			const Eigen::Affine3f T(absolute_transforms[nid]);
			const auto &point_set = point_sets[nid];
			const auto &Tcaminit = std::get<3>(point_set);

			auto points_viz(std::make_shared<points_type>());
			pcl::transformPointCloud(*std::get<0>(point_set), *points_viz, T);  // Raw point cloud.
			//pcl::transformPointCloud(*std::get<1>(point_set), *points_viz, T);  // Downsampled point cloud.

			const std::string cloud_id("Point Cloud #" + std::to_string(nid)), normal_id("Point Cloud Normal #" + std::to_string(nid)), camera_id("Camera #" + std::to_string(nid));
			viewer.addPointCloud<point_type>(points_viz, cloud_id);
			//viewer.addPointCloudNormals<point_type, normal_type>(std::get<0>(point_set), std::get<2>(point_set), 1, 5.0f, normal_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, cloud_id);
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, cloud_id);
			viewer.addCoordinateSystem(50.0, T * Tcaminit, camera_id);
		}

		//viewer.setRepresentationToSurfaceForAllActors();
		viewer.resetCamera();

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			//std::this_thread::sleep_for(50ms);
		}
	}

	if (visualize)
	{
		pcl::visualization::PCLVisualizer viewer("Camera & Landmark Viewer before BA");
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

		auto cloud(std::make_shared<points_type>());
		cloud->reserve(pg_landmarks.size());
		std::transform(pg_landmarks.begin(), pg_landmarks.end(), std::back_inserter(cloud->points), [](const auto &l) { return point_type(float(l.second[0]), float(l.second[1]), float(l.second[2])); });

		viewer.addPointCloud<point_type>(cloud, "Point Cloud");
		//viewer.addPointCloudNormals<point_type, normal_type>(cloud, cloud_normal, 1, 5.0f, "Point Cloud Normal");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Point Cloud");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Point Cloud");
		//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "Point Cloud");

		for (const auto &nid: pose_graph)
		{
			const Eigen::Affine3f T(absolute_transforms[nid]);
			const auto &point_set = point_sets[nid];
			const auto &Tcaminit = std::get<3>(point_set);

			viewer.addCoordinateSystem(50.0, T * Tcaminit, "Camera #" + std::to_string(nid));
		}

		//viewer.setRepresentationToSurfaceForAllActors();
		viewer.resetCamera();

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			//std::this_thread::sleep_for(50ms);
		}
	}

	// Bundle adjustment (BA).
	// Refer to slam2d_tutorial(), ba_example(), slam3d_se3_pointxyz_test(), & slam3d_se3_test().
	{
		std::cout << "Performing BA (total)..." << std::endl;
		const auto start_time_total(std::chrono::high_resolution_clock::now());

		const bool STRUCTURE_ONLY = false;
		const bool ROBUST_KERNEL = false;
		const bool DENSE = false;

		g2o::SparseOptimizer optimizer;
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
			linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType> >();
			std::cout << "Using DENSE" << std::endl;
		}
		else
		{
#if defined(G2O_HAVE_CHOLMOD)
			linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> >();
			std::cout << "Using CHOLMOD" << std::endl;
#else
			linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType> >();
			std::cout << "Using Eigen" << std::endl;
#endif
		}
		//auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
		auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
		optimizer.setAlgorithm(solver);
#endif

#if !defined(__USE_SE3_AS_LANDMARKS)
		// Add the parameter representing the sensor offset.
		// FIXME [check] >> Camera parameters.
#if 0
		const double focal_length = 1000.0;
		const g2o::Vector2 principal_point(150.0, 225.0);  // For 300x450 images.
		const double baseline = 0.0;
		auto camera_param = new g2o::CameraParameters(focal_length, principal_point, baseline);
#elif 0
		const double fx = 1000.0, fy = 1000.0;
		const double cx = 150.0, cy = 225.0;  // For 300x450 images.
		auto camera_param = new g2o::ParameterCamera();
		camera_param->setKcam(fx, fy, cx, cy);
#else
		auto sensorOffsetTransf(g2o::Isometry3::Identity());
		//sensorOffsetTransf(3, 3) = 100.0;
		auto camera_param = new g2o::ParameterSE3Offset;
		camera_param->setOffset(sensorOffsetTransf);
#endif
		camera_param->setId(0);

		if (!optimizer.addParameter(camera_param))
		{
			assert(false);
		}
#endif

		// Construct a pose graph.
		{
			std::cout << "Constructing a pose graph..." << std::endl;
			const auto start_time(std::chrono::high_resolution_clock::now());

			// Add the camera poses (vertices).
			std::cout << "Adding camera poses..." << std::endl;
			for (const auto &nid: pose_graph)
			{
				const Eigen::Affine3f T(absolute_transforms[nid]);
				const auto &point_set = point_sets[nid];
				const auto &Tcaminit = std::get<3>(point_set);
				// FIXME [check] >> Camera pose.
				const auto Tcam(T * Tcaminit);
				//const auto &Tcam = T;

				auto camera = new g2o::VertexSE3;
				camera->setId(int(nid));
				//camera->setEstimate(g2o::SE3Quat(Tcam.rotation().cast<double>(), Tcam.translation().cast<double>()));  // Camera pose wrt the world frame.
				camera->setEstimate(g2o::Isometry3(Tcam.matrix().cast<double>()));  // Camera pose wrt the world frame.
				//camera->setMarginalized(true);
				//camera->setFixed(true);

				if (!optimizer.addVertex(camera))
					std::cerr << "Camera vertex #" << nid << " not added." << std::endl;
			}
			std::cout << "Done." << std::endl;

			// Add the landmarks (vertices).
			std::cout << "Adding landmarks..." << std::endl;
			for (const auto &l: pg_landmarks)
			{
#if defined(__USE_SE3_AS_LANDMARKS)
				auto landmark = new g2o::VertexSE3;
				landmark->setId(int(l.first));
				landmark->setEstimate(g2o::Isometry3(Eigen::Translation3d(l.second) * Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0)));  // Vertex pose wrt the world frame.
#else
				auto landmark = new g2o::VertexPointXYZ;
				landmark->setId(int(l.first));
				landmark->setEstimate(l.second);  // Vertex coordinates wrt the world frame.
#endif
				landmark->setMarginalized(true);  // If true, faster.

				if (!optimizer.addVertex(landmark))
					std::cerr << "Landmark vertex #" << l.first << " not added." << std::endl;
			}
			std::cout << "Done." << std::endl;
			assert(optimizer.vertices().size() == point_sets.size() + pg_landmarks.size());

			// Add the landmark constraints (edges).
			{
				std::cout << "Adding landmark observations..." << std::endl;
				// FIXME [check] >> Information matrix.
#if defined(__USE_SE3_AS_LANDMARKS)
				const Eigen::Matrix<double, 6, 6> information_matrix(Eigen::DiagonalMatrix<double, 6>(0.2, 0.2, 0.2, 0.000001, 0.000001, 0.000001));
#else
				const auto information_matrix(Eigen::Matrix3d::Identity() * 0.2);
				//const Eigen::DiagonalMatrix<double, 3> information_matrix(0.2, 0.2, 0.2);
#endif
				for (const auto &nid: pose_graph)
				{
					const Eigen::Affine3f T(absolute_transforms[nid]);
					const auto &point_set = point_sets[nid];
					const auto &cloud = std::get<0>(point_set);  // Raw point cloud.
					//const auto &cloud = std::get<1>(point_set);  // Downsampled point cloud.
					const auto &Tcaminit = std::get<3>(point_set);  // The initial camera pose wrt the world frame.
					const auto &point_id_to_landmark_id_mapper = point_id_to_pg_landmark_id_mappers[nid];

					for (size_t pidx = 0; pidx < cloud->size(); ++pidx)
					{
						const auto &pt = cloud->points[pidx];  // Initial coordinates wrt the world frame.
						const Eigen::Vector3f pt_w(pt.x, pt.y, pt.z);  // Relative coordinates of each landmark wrt the world frame.
						//const auto pt_c(Tcaminit.inverse() * pt_w);  // Error. Relative coordinates of each landmark wrt the camera frame.
						const Eigen::Vector3f pt_c(Tcaminit.inverse() * pt_w);  // Relative coordinates of each landmark wrt the camera frame.
#if defined(__USE_SE3_AS_LANDMARKS)
						const g2o::Isometry3 measurement(Eigen::Translation3d(pt_c.cast<double>()) * Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));  // Relative pose of each vertex wrt the world frame.

						auto observation = new g2o::EdgeSE3;
#else
						const auto measurement(pt_c.cast<double>());  // Relative coordinates of each landmark wrt the camera frame.

						auto observation = new g2o::EdgeSE3PointXYZ;
#endif
						observation->vertices()[0] = optimizer.vertex(int(nid));  // Camera ID.
						observation->vertices()[1] = optimizer.vertex(int(point_id_to_landmark_id_mapper[pidx]));  // Landmark ID.
						observation->setMeasurement(measurement);
						observation->setInformation(information_matrix);
#if !defined(__USE_SE3_AS_LANDMARKS)
						observation->setParameterId(0, camera_param->id());
#endif
						if (ROBUST_KERNEL)
						{
							//auto robust_kernel = g2o::RobustKernelFactory::instance()->construct("Huber");
							auto robust_kernel = new g2o::RobustKernelHuber;
							robust_kernel->setDelta(std::sqrt(5.991));  // 95% CI.
							observation->setRobustKernel(robust_kernel);
						}

						if (!optimizer.addEdge(observation))
							std::cerr << "Landmark observation edge(" << nid << " - " << point_id_to_landmark_id_mapper[pidx] << ") not added." << std::endl;
					}
				}
				std::cout << "Done." << std::endl;
			}

			std::cout << "A pose graph constructed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
			std::cout << "Pose graph:" << std::endl;
			std::cout << "\t#vertices = " << optimizer.vertices().size() << ", #edges = " << optimizer.edges().size() << ", #parameters = " << optimizer.parameters().size() << std::endl;
			std::cout << "\tMax dimension = " << optimizer.maxDimension() << std::endl;
			std::cout << "\tChi2 = " << optimizer.chi2() << std::endl;
		}

		// Optimization.
		{
			// Prepare and run the optimization.
			// Fix the first camera pose to account for gauge freedom.
			auto firstCameraPose = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(pose_graph[0]));
			firstCameraPose->setFixed(true);

			const int max_iterations = 100;
			const bool online = false;
			const bool verbose = true;

			optimizer.setVerbose(verbose);
			if (optimizer.initializeOptimization())
			{
				// Dump initial state to the disk.
				//if (!optimizer.save("../ba_before.g2o"))
				//	std::cerr << "Failed to save to " << "../ba_before.g2o" << std::endl;

				if (STRUCTURE_ONLY)
				{
					const int max_trials = 10;

					std::cout << "Performing structure-only BA:" << std::endl;
					const auto start_time(std::chrono::high_resolution_clock::now());
					g2o::StructureOnlySolver<3> structure_only_ba;
					g2o::OptimizableGraph::VertexContainer points;
					for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
					{
						auto v = static_cast<g2o::OptimizableGraph::Vertex *>(it->second);
						if (v->dimension() == 3)  points.push_back(v);
					}
					const auto &solver_result = structure_only_ba.calc(points, max_iterations, max_trials);
					std::cout << "Structure-only BA performed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
					std::cout << "Structure-only BA solver result: " << solver_result << std::endl;
				}

				//if (!optimizer.save("../structure_only_ba_after.g2o"))
				//	std::cerr << "Failed to save to " << "../ba_after_structure_only.g2o" << std::endl;

				{
					std::cout << "Performing full BA..." << std::endl;
					const auto start_time(std::chrono::high_resolution_clock::now());
					const auto num_iterations = optimizer.optimize(max_iterations, online);
					if (num_iterations > 0)
						std::cout << "Full BA performed (#iterations = " << num_iterations << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
					else
						std::cout << "Full BA failed." << std::endl;
				}

				//if (!optimizer.save("../ba_after.g2o"))
				//	std::cerr << "Failed to save to " << "../ba_after.g2o" << std::endl;
			}
			else
				std::cerr << "Optimization not initialized." << std::endl;
		}

		// Show BA results.
		{
			// Cameras.
			const size_t num_cameras_sampled = 10;
			std::vector<size_t> pose_graph_sampled;
			std::sample(pose_graph.begin(), pose_graph.end(), std::back_inserter(pose_graph_sampled), num_cameras_sampled, std::mt19937 { std::random_device{}() });
			for (const auto &nid: pose_graph_sampled)
			{
				const Eigen::Affine3f T(absolute_transforms[nid]);
				const auto &point_set = point_sets[nid];
				const auto &Tcaminit = std::get<3>(point_set);
				const auto Tcam(T * Tcaminit);

				const g2o::Isometry3 &Tcam_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(nid)))->estimate();

				std::cout << "Camera ID #" << nid << " ----------" << std::endl;
				std::cout << Tcam.matrix() << std::endl;
				std::cout << Tcam_est.matrix() << std::endl;
			}

			// Landmarks.
			const size_t num_landmarks_sampled = 10;
			std::map<size_t, g2o::Vector3> pg_landmarks_sampled;
			std::sample(pg_landmarks.begin(), pg_landmarks.end(), std::inserter(pg_landmarks_sampled, pg_landmarks_sampled.end()), num_landmarks_sampled, std::mt19937 { std::random_device{}() });
			for (const auto &l: pg_landmarks_sampled)
			{
#if defined(__USE_SE3_AS_LANDMARKS)
				const auto &l_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(l.first)))->estimate();

				std::cout << "Landmark ID #" << l.first << " ----------" << std::endl;
				std::cout << l.second.transpose() << std::endl;
				std::cout << l_est.matrix() << std::endl;
#else
				const auto &l_est = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(int(l.first)))->estimate();

				std::cout << "Landmark ID #" << l.first << " ----------" << std::endl;
				std::cout << l.second.transpose() << std::endl;
				std::cout << l_est.transpose() << std::endl;
#endif
			}
		}

		// Visualize.
		if (visualize)
		{
			pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer after BA");
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

			for (const auto &nid: pose_graph)
			{
				//const Eigen::Affine3f T(absolute_transforms[nid]);
				const auto &point_set = point_sets[nid];
				const auto &Tcaminit = std::get<3>(point_set);

				//const auto Tcam(T * Tcaminit);
				const Eigen::Affine3f Tcam_est(dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(nid)))->estimate().cast<float>());
				const auto T(Tcam_est * Tcaminit.inverse());

				auto points_viz(std::make_shared<points_type>());
				pcl::transformPointCloud(*std::get<0>(point_set), *points_viz, T);  // Raw point cloud.
				//pcl::transformPointCloud(*std::get<1>(point_set), *points_viz, T);  // Downsampled point cloud.

				const std::string cloud_id("Point Cloud #" + std::to_string(nid)), normal_id("Point Cloud Normal #" + std::to_string(nid)), camera_id("Camera #" + std::to_string(nid));
				viewer.addPointCloud<point_type>(points_viz, cloud_id);
				//viewer.addPointCloudNormals<point_type, normal_type>(std::get<0>(point_set), std::get<2>(point_set), 1, 5.0f, normal_id);
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id);
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, (std::rand() % 101) * 0.01, cloud_id);
				//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, cloud_id);
				viewer.addCoordinateSystem(50.0, Tcam_est, camera_id);
			}

			//viewer.setRepresentationToSurfaceForAllActors();
			viewer.resetCamera();

			while (!viewer.wasStopped())
			{
				viewer.spinOnce();
				//std::this_thread::sleep_for(50ms);
			}
		}

		if (visualize)
		{
			pcl::visualization::PCLVisualizer viewer("Camera & Landmark Viewer after BA");
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

			auto cloud(std::make_shared<points_type>());
			cloud->reserve(pg_landmarks.size());
			for (const auto &l: pg_landmarks)
			{
#if defined(__USE_SE3_AS_LANDMARKS)
				//const auto &l_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(l.first)))->estimate();
				const g2o::Vector3 l_est(dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(l.first)))->estimate().translation());
#else
				const auto &l_est = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(int(l.first)))->estimate();
#endif
				cloud->push_back(point_type(float(l_est.x()), float(l_est.y()), float(l_est.z())));
			}

			viewer.addPointCloud<point_type>(cloud, "Point Cloud");
			//viewer.addPointCloudNormals<point_type, normal_type>(cloud, cloud_normal, 1, 5.0f, "Point Cloud Normal");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Point Cloud");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Point Cloud");
			//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_SHADING, pcl::visualization::PCL_VISUALIZER_SHADING_PHONG, "Point Cloud");

			for (const auto &nid: pose_graph)
			{
				const g2o::Isometry3 &Tcam_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(int(nid)))->estimate();
				viewer.addCoordinateSystem(50.0, Eigen::Affine3f(Tcam_est.matrix().cast<float>()), "Camera #" + std::to_string(nid));
			}

			//viewer.setRepresentationToSurfaceForAllActors();
			viewer.resetCamera();

			while (!viewer.wasStopped())
			{
				viewer.spinOnce();
				//std::this_thread::sleep_for(50ms);
			}
		}

		// Free the graph memory.
		optimizer.clear();

		std::cout << "BA performed (total): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_total).count() << " msecs." << std::endl;
	}
}

}  // namespace my_g2o
