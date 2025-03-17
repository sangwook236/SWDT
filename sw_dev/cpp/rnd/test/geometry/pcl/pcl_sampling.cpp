#include <chrono>
#include <iterator>
#include <algorithm>
#include <string>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/farthest_point_sampling.h>
#include <pcl/filters/normal_space.h>
#include <pcl/filters/covariance_sampling.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>


using namespace std::literals::chrono_literals;

namespace {
namespace local {

// REF [site] >> http://www.pointclouds.org/documentation/tutorials/resampling.php
void resampling_tutorial()
{
	// Load input file into a PointCloud<T> with an appropriate type.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	// Load bun0.pcd -- should be available with the PCL archive in test.
	pcl::io::loadPCDFile("./data/geometry/pcl/bun0.pcd", *cloud);

	// Create a KD-Tree.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	// Output has the PointNormal type in order to store the normals calculated by MLS.
	pcl::PointCloud<pcl::PointNormal> mls_points;

	// Init object (second point type is for the normals, even if unused).
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

	mls.setComputeNormals(true);

	// Set parameters.
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.03);

	// Reconstruct.
	mls.process(mls_points);

	// Save output.
	pcl::io::savePCDFile("./data/geometry/pcl/bun0_mls.pcd", mls_points);
}

void subsampling_test()
{
	using PointType = pcl::PointXYZ;

	//const std::string input_filepath("/path/to/input.pcd");
	const std::string input_filepath("../data/bunny.ply");
	//const std::string input_filepath("../data/horse.ply");

	// Load a point cloud
	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
	{
		std::cout << "Loading a point cloud..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//if (pcl::io::loadPCDFile(input_filepath, *cloud) == -1)
		if (pcl::io::loadPLYFile(input_filepath, *cloud) == -1)
		{
			const std::string err("File not found, " + input_filepath + ".\n");
			PCL_ERROR(err.c_str());
			return;
		}
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "A point cloud loaded (" << cloud->size() << " points, " << pcl::getFieldsList(*cloud) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	const size_t num_points_to_sample(cloud->size() / 5);
	const float grid_leaf_size(0.01f);
	const double normal_search_radius(0.03);
	const unsigned int num_bins_for_NSS(4);
	//const double curvature_search_radius(0.03);
	const double curvature_search_radius(0.003);  // Faster and more locally
	const float invalid_curvature_threshold(1.0e-5f);
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);

	// Estimate normals
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	{
		std::cout << "Estimating normals..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		//pcl::NormalEstimation<PointType, pcl::Normal> normal_estimation;
		pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_estimation;
		//pcl::gpu::NormalEstimation normal_estimation;  // Only for pcl::PointXYZ
		normal_estimation.setRadiusSearch(normal_search_radius);
		//normal_estimation.setKSearch(normal_search_k);
		normal_estimation.setSearchMethod(tree);
		normal_estimation.setInputCloud(cloud);
		normal_estimation.compute(*normals);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Normals estimated: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
	}

	{
		std::cout << "Downsampling using VoxelGrid..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_voxelgrid(new pcl::PointCloud<PointType>);
		pcl::VoxelGrid<PointType> sor;
		sor.setLeafSize(grid_leaf_size, grid_leaf_size, grid_leaf_size);
		sor.setInputCloud(cloud);
		sor.filter(*cloud_voxelgrid);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Downsampled using VoxelGrid (" << cloud_voxelgrid->size() << " points, " << pcl::getFieldsList(*cloud_voxelgrid) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_voxelgrid.pcd", *cloud_voxelgrid);
		pcl::io::savePLYFile("./subsampling_voxelgrid.ply", *cloud_voxelgrid);
	}

	{
		// Uniform sampling
		//	Good for maintaining point density
		std::cout << "Uniform subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_uniform(new pcl::PointCloud<PointType>);
		pcl::UniformSampling<PointType> uniform_sampling;
		uniform_sampling.setRadiusSearch(grid_leaf_size);
		uniform_sampling.setInputCloud(cloud);
		uniform_sampling.filter(*cloud_uniform);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Uniform subsampling done (" << cloud_uniform->size() << " points, " << pcl::getFieldsList(*cloud_uniform) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_uniform.pcd", *cloud_uniform);
		pcl::io::savePLYFile("./subsampling_uniform.ply", *cloud_uniform);
	}

	{
		// Random sampling
		//	Fast and simple
		std::cout << "Random subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_random(new pcl::PointCloud<PointType>);
		pcl::RandomSample<PointType> random_sample;
		random_sample.setSample((unsigned int)num_points_to_sample);
		random_sample.setInputCloud(cloud);
		random_sample.filter(*cloud_random);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Random subsampling done (" << cloud_random->size() << " points, " << pcl::getFieldsList(*cloud_random) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_random.pcd", *cloud_random);
		pcl::io::savePLYFile("./subsampling_random.ply", *cloud_random);
	}

	{
		// Farthest point sampling (FPS)
		//	Good for maintaining geometric features
		std::cout << "Farthest point subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_farthest(new pcl::PointCloud<PointType>);
		pcl::FarthestPointSampling<PointType> farthest_point_sampling;
		farthest_point_sampling.setSample(num_points_to_sample);
		farthest_point_sampling.setInputCloud(cloud);
		farthest_point_sampling.filter(*cloud_farthest);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Farthest point subsampling done (" << cloud_farthest->size() << " points, " << pcl::getFieldsList(*cloud_farthest) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_farthest.pcd", *cloud_farthest);
		pcl::io::savePLYFile("./subsampling_farthest.ply", *cloud_farthest);
	}

	{
		// Normal-space sampling (NSS)
		//	Good for maintaining surface variation
		std::cout << "Normal-space subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_normal_space(new pcl::PointCloud<PointType>);
		pcl::NormalSpaceSampling<PointType, pcl::Normal> normal_space_sampling;
		normal_space_sampling.setBins(num_bins_for_NSS, num_bins_for_NSS, num_bins_for_NSS);
		normal_space_sampling.setSample((unsigned int)num_points_to_sample);
		normal_space_sampling.setInputCloud(cloud);
		normal_space_sampling.setNormals(normals);
		normal_space_sampling.filter(*cloud_normal_space);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Normal-space subsampling done (" << cloud_normal_space->size() << " points, " << pcl::getFieldsList(*cloud_normal_space) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_normal_space.pcd", *cloud_normal_space);
		pcl::io::savePLYFile("./subsampling_normal_space.ply", *cloud_normal_space);
	}

	{
		// Covariance sampling
		//	Good for maintaining important shape features
		std::cout << "Covariance subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<PointType>::Ptr cloud_covariance(new pcl::PointCloud<PointType>);
		pcl::CovarianceSampling<PointType, pcl::Normal> covariance_sampling;
		covariance_sampling.setNumberOfSamples((unsigned int)num_points_to_sample);
		covariance_sampling.setInputCloud(cloud);
		covariance_sampling.setNormals(normals);
		covariance_sampling.filter(*cloud_covariance);
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		std::cout << "Covariance subsampling done (" << cloud_covariance->size() << " points, " << pcl::getFieldsList(*cloud_covariance) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_covariance.pcd", *cloud_covariance);
		pcl::io::savePLYFile("./subsampling_covariance.ply", *cloud_covariance);
	}

	{
		// Curvature sampling
		//	Good for maintaining areas of high surface variation
		std::cout << "Curvature subsampling..." << std::endl;
		const auto start_time(std::chrono::high_resolution_clock::now());
		pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
		pcl::PrincipalCurvaturesEstimation<PointType, pcl::Normal, pcl::PrincipalCurvatures> curvature_estimation;
		//pcl::gpu::PrincipalCurvaturesEstimation curvature_estimation;  // Only for pcl::PointXYZ
		curvature_estimation.setRadiusSearch(curvature_search_radius);
		//curvature_estimation.setKSearch(curvature_search_k);
		curvature_estimation.setSearchMethod(tree);
		curvature_estimation.setInputCloud(cloud);
		curvature_estimation.setInputNormals(normals);
		curvature_estimation.compute(*curvatures);

		const auto num_curvatures(curvatures->size());
#if 0
		std::vector<size_t> curvature_indices(num_curvatures);
		//std::generate(curvature_indices.begin(), curvature_indices.end(), [n = 0]() mutable { return n++; });
		std::iota(curvature_indices.begin(), curvature_indices.end(), 0);
#elif 0
		std::vector<size_t> curvature_indices;
		curvature_indices.reserve(num_curvatures);
		std::generate_n(std::back_inserter(curvature_indices), num_curvatures, [n = 0]() mutable { return n++; });
#else
		// NOTE [caution] >> std::numeric_limits<float>::quiet_NaN() can be treated as 0 during operations
		std::vector<size_t> curvature_indices;
		curvature_indices.reserve(num_curvatures);
		std::for_each(curvatures->begin(), curvatures->end(), [&curvature_indices, idx = 0](const auto curvature) mutable {
			//if (!std::isnan(curvature.pc1) && !std::isnan(curvature.pc2))
			if (!std::isnan(curvature.pc1))
			curvature_indices.push_back(idx);
			++idx;
		});
#endif
		//auto curvature_threshold_index_it = curvature_indices.begin() + std::max((int)num_curvatures - (int)num_points_to_sample, 0);
		auto curvature_threshold_index_it = curvature_indices.begin() + std::max((int)curvature_indices.size() - (int)num_points_to_sample, 0);
		std::nth_element(curvature_indices.begin(), curvature_threshold_index_it, curvature_indices.end(), [&curvatures](const auto &lhs, const auto &rhs) {
			//return std::abs(curvatures->points[lhs].pc1) < std::abs(curvatures->points[rhs].pc1) && std::abs(curvatures->points[lhs].pc2) < std::abs(curvatures->points[rhs].pc2);
			return std::abs(curvatures->points[lhs].pc1) < std::abs(curvatures->points[rhs].pc1);
		});
		//if (std::min(std::abs(curvatures->points[*curvature_threshold_index_it].pc1), std::abs(curvatures->points[*curvature_threshold_index_it].pc2)) < invalid_curvature_threshold)
		if (std::abs(curvatures->points[*curvature_threshold_index_it].pc1) < invalid_curvature_threshold)
		{
			// Find the minimum larger than invalid_curvature_threshold in order to use all points with curvature larger than invalid_curvature_threshold
			curvature_threshold_index_it = std::min_element(curvature_threshold_index_it, curvature_indices.end(), [&curvatures, invalid_curvature_threshold](const auto &lhs, const auto &rhs) {
				const auto lhs_pc1(std::abs(curvatures->points[lhs].pc1));
				const auto rhs_pc1(std::abs(curvatures->points[rhs].pc1));
				if (lhs_pc1 < invalid_curvature_threshold && rhs_pc1 < invalid_curvature_threshold)
					return lhs_pc1 < rhs_pc1;
				else if (lhs_pc1 < invalid_curvature_threshold) return false;
				else if (rhs_pc1 < invalid_curvature_threshold) return true;
				else return lhs_pc1 < rhs_pc1;
			});
			if (std::abs(curvatures->points[*curvature_threshold_index_it].pc1) < invalid_curvature_threshold)
				curvature_threshold_index_it = curvature_indices.end();
		}
		//const float curvature_threshold(std::min(curvature_indices.end() != curvature_threshold_index_it ? std::abs(curvatures->points[*curvature_threshold_index_it].pc1), std::abs(curvatures->points[*curvature_threshold_index_it].pc2)) : std::numeric_limits<float>::quiet_NaN());
		const float curvature_threshold(curvature_indices.end() != curvature_threshold_index_it ? std::abs(curvatures->points[*curvature_threshold_index_it].pc1) : std::numeric_limits<float>::quiet_NaN());
		pcl::PointCloud<PointType>::Ptr cloud_curvature(new pcl::PointCloud<PointType>);
		cloud_curvature->reserve(num_curvatures);
		std::for_each(curvature_threshold_index_it, curvature_indices.end(), [&cloud, &cloud_curvature](const auto &idx) {
			cloud_curvature->push_back(cloud->points[idx]);
		});
#if 0
		// TODO [check] >> Is it necessary to filter curvatures by pc2?
		const float curvature_threshold_pc2(curvature_threshold * 0.5f);
		std::for_each(curvature_indices.begin(), curvature_threshold_index_it, [&cloud, &curvatures, &cloud_curvature, pc2_thres = curvature_threshold_pc2](const auto &idx) {
			// NOTE [info] >> std::abs(NaN) = NaN
			//const auto pc1(std::abs(curvatures->points[idx].pc1));
			const auto pc2(std::abs(curvatures->points[idx].pc2));
			if (std::isnan(pc2)) return;
			//if (pc2 / pc1 > 0.75f)
			if (pc2 > pc2_thres)
				cloud_curvature->push_back(cloud->points[idx]);
		});
#endif
		const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
		//std::cout << "Curvature subsampling done (" << num_curvatures << " points, " << pcl::getFieldsList(*curvatures) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
		std::cout << "Curvature subsampling done (" << cloud_curvature->size() << " points, " << pcl::getFieldsList(*cloud_curvature) << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;
		std::cout << "\tCurvature threshold = " << curvature_threshold << std::endl;

		// Save the results
		//pcl::io::savePCDFile("./subsampling_curvature.pcd", *cloud_curvature);
		pcl::io::savePLYFile("./subsampling_curvature.ply", *cloud_curvature);

		//-----
		// Check validity
		{
			// If negative curvatures
			const auto num_negative_curvatures_pc1 = std::count_if(curvatures->begin(), curvatures->end(), [](const auto &curvature) {
				return curvature.pc1 < 0;
			});
			std::cout << "#negative curvatures in pc1 = " << num_negative_curvatures_pc1 << std::endl;
			const auto num_negative_curvatures_pc2 = std::count_if(curvatures->begin(), curvatures->end(), [](const auto &curvature) {
				return curvature.pc2 < 0;
			});
			std::cout << "#negative curvatures in pc2 = " << num_negative_curvatures_pc2 << std::endl;

			// If pc1 < pc2
			const auto num_curvatures_pc1_pc2 = std::count_if(curvatures->begin(), curvatures->end(), [](const auto &curvature) {
				return curvature.pc1 < curvature.pc2;
			});
			std::cout << "#curvatures of pc1 < pc2 = " << num_curvatures_pc1_pc2 << std::endl;

			// If abs(pc1) < abs(pc2)
			const auto num_curvatures_pc1_pc2_abs = std::count_if(curvatures->begin(), curvatures->end(), [](const auto &curvature) {
				return std::abs(curvature.pc1) < std::abs(curvature.pc2);
			});
			std::cout << "#curvatures of abs(pc1) < abs(pc2) = " << num_curvatures_pc1_pc2_abs << std::endl;

			// If NaN
			const auto num_curvatures_with_nan = std::count_if(curvatures->begin(), curvatures->end(), [](const auto &curvature) {
				return std::isnan(curvature.pc1) || std::isnan(curvature.pc2);
				//return std::isnan(curvature.pc1);
			});
			std::cout << "#curvatures with NaN = " << num_curvatures_with_nan << std::endl;
		}

		// Check computation time
		{
			const size_t &num_samples = num_points_to_sample;
			//const size_t num_samples(num_curvatures / 2);
			const size_t iterations(1000);

			{
				std::cout << "Using separate indices..." << std::endl;
				const auto start_time(std::chrono::high_resolution_clock::now());
				for (size_t iter = 0; iter < iterations; ++iter)
				{
#if 0
					std::vector<size_t> indices(num_curvatures);
					//std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
					std::iota(indices.begin(), indices.end(), 0);
#elif 0
					std::vector<size_t> indices;
					indices.reserve(num_curvatures);
					std::generate_n(std::back_inserter(indices), num_curvatures, [n = 0]() mutable { return n++; });
#else
					// NOTE [caution] >> std::numeric_limits<float>::quiet_NaN() can be treated as 0 during operations
					std::vector<size_t> indices;
					indices.reserve(num_curvatures);
					std::for_each(curvatures->begin(), curvatures->end(), [&indices, idx = 0](const auto curvature) mutable {
						if (!std::isnan(curvature.pc1))
							indices.push_back(idx);
						++idx;
					});
#endif
					//auto threshold_it1 = indices.begin() + (num_curvatures - num_samples);
					auto threshold_it1 = indices.begin() + std::max((int)indices.size() - (int)num_samples, 0);
					std::nth_element(indices.begin(), threshold_it1, indices.end(), [&curvatures](const auto &lhs, const auto &rhs) {
						return curvatures->points[lhs].pc1 < curvatures->points[rhs].pc1;
					});

#if 0
					//std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
					std::iota(indices.begin(), indices.end(), 0);
#elif 0
					indices.clear();
					std::generate_n(std::back_inserter(indices), num_curvatures, [n = 0]() mutable { return n++; });
#else
					// NOTE [caution] >> std::numeric_limits<float>::quiet_NaN() can be treated as 0 during operations
					indices.clear();
					std::for_each(curvatures->begin(), curvatures->end(), [&indices, idx = 0](const auto curvature) mutable {
						if (!std::isnan(curvature.pc1))
							indices.push_back(idx);
						++idx;
					});
#endif
					//auto threshold_it2 = indices.begin() + (num_curvatures - num_samples);
					auto threshold_it2 = indices.begin() + std::max((int)indices.size() - (int)num_samples, 0);
					std::nth_element(indices.begin(), threshold_it2, indices.end(), [&curvatures](const auto &lhs, const auto &rhs) {
						return curvatures->points[lhs].pc2 < curvatures->points[rhs].pc2;
					});

#if 1
					pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
					cloud_filtered->reserve(num_samples);
					std::for_each(threshold_it2, indices.end(), [&cloud, &cloud_filtered](const auto &idx) {
						cloud_filtered->push_back(cloud->points[idx]);
					});
#endif
				}
				const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
				std::cout << "Separate indices used (#iterations = " << iterations << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

				{
#if 0
					std::vector<size_t> indices(num_curvatures);
					//std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
					std::iota(indices.begin(), indices.end(), 0);
#elif 0
					std::vector<size_t> indices;
					indices.reserve(num_curvatures);
					std::generate_n(std::back_inserter(indices), num_curvatures, [n = 0]() mutable { return n++; });
#else
					// NOTE [caution] >> std::numeric_limits<float>::quiet_NaN() can be treated as 0 during operations
					std::vector<size_t> indices;
					indices.reserve(num_curvatures);
					std::for_each(curvatures->begin(), curvatures->end(), [&indices, idx = 0](const auto curvature) mutable {
						if (!std::isnan(curvature.pc1))
							indices.push_back(idx);
						++idx;
					});
#endif
					//auto threshold_it1 = indices.begin() + (num_curvatures - num_samples);
					auto threshold_it1 = indices.begin() + std::max((int)indices.size() - (int)num_samples, 0);
					std::nth_element(indices.begin(), threshold_it1, indices.end(), [&curvatures](const auto &lhs, const auto &rhs) {
						return curvatures->points[lhs].pc1 < curvatures->points[rhs].pc1;
					});
					std::cout << "\t#points to sample by pc1 = " << std::distance(threshold_it1, indices.end()) << std::endl;
					std::cout << "\tThreshold of pc1 = " << curvatures->points[*threshold_it1].pc1 << std::endl;
					std::cout << "\t";
					std::for_each(threshold_it1, threshold_it1 + 10, [&curvatures](const auto idx) {
						std::cout << curvatures->points[idx].pc1 << ", ";
					});
					std::cout << std::endl;

#if 0
					//std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
					std::iota(indices.begin(), indices.end(), 0);
#elif 0
					indices.clear();
					std::generate_n(std::back_inserter(indices), num_curvatures, [n = 0]() mutable { return n++; });
#else
					// NOTE [caution] >> std::numeric_limits<float>::quiet_NaN() can be treated as 0 during operations
					indices.clear();
					std::for_each(curvatures->begin(), curvatures->end(), [&indices, idx = 0](const auto curvature) mutable {
						if (!std::isnan(curvature.pc1))
							indices.push_back(idx);
						++idx;
					});
#endif
					//auto threshold_it2 = indices.begin() + (num_curvatures - num_samples);
					auto threshold_it2 = indices.begin() + std::max((int)indices.size() - (int)num_samples, 0);
					std::nth_element(indices.begin(), threshold_it2, indices.end(), [&curvatures](const auto &lhs, const auto &rhs) {
						return curvatures->points[lhs].pc2 < curvatures->points[rhs].pc2;
					});
					std::cout << std::endl;
					std::cout << "\t#points to sample by pc2 = " << std::distance(threshold_it2, indices.end()) << std::endl;
					std::cout << "\tThreshold of pc2 = " << curvatures->points[*threshold_it2].pc2 << std::endl;
					std::cout << "\t";
					std::for_each(threshold_it2, threshold_it2 + 10, [&curvatures](const auto idx) {
						std::cout << curvatures->points[idx].pc2 << ", ";
					});
					std::cout << std::endl;
				}
			}
			{
				std::cout << "Using duplicate curvatures..." << std::endl;
				const auto start_time(std::chrono::high_resolution_clock::now());
				for (size_t iter = 0; iter < iterations; ++iter)
				{
					std::vector<float> pc1s, pc2s;
					pc1s.reserve(num_curvatures);
					pc2s.reserve(num_curvatures);
					std::for_each(curvatures->begin(), curvatures->end(), [&pc1s, &pc2s](const auto &c) {
						if (!std::isnan(c.pc1)) pc1s.push_back(c.pc1);
						if (!std::isnan(c.pc2)) pc2s.push_back(c.pc2);
					});

					//auto threshold_it1 = pc1s.begin() + (num_curvatures - num_samples);
					auto threshold_it1 = pc1s.begin() + std::max((int)pc1s.size() - (int)num_samples, 0);
					std::nth_element(pc1s.begin(), threshold_it1, pc1s.end());

					//auto threshold_it2 = pc2s.begin() + (num_curvatures - num_samples);
					auto threshold_it2 = pc2s.begin() + std::max((int)pc2s.size() - (int)num_samples, 0);
					std::nth_element(pc2s.begin(), threshold_it2, pc2s.end());

#if 1
					pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
					cloud_filtered->reserve(num_samples);
					//std::for_each(curvatures->begin(), curvatures->end(), [&cloud, &cloud_filtered, pc1_thresh = *threshold_it1, pc2_thresh = *threshold_it2](const auto &c) {
					//	if (c.pc1 > pc1_thresh || c.pc2 > pc2_thresh)
					//		cloud_filtered->push_back(cloud->points[idx]);
					//});
					for (size_t idx = 0; idx < num_curvatures; ++idx)
					{
						if (curvatures->points[idx].pc1 > *threshold_it1 || curvatures->points[idx].pc2 > *threshold_it2)
							cloud_filtered->push_back(cloud->points[idx]);
					}
#endif
				}
				const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
				std::cout << "Duplicate curvatures used (#iterations = " << iterations << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " msecs." << std::endl;

				{
					std::vector<float> pc1s, pc2s;
					pc1s.reserve(num_curvatures);
					pc2s.reserve(num_curvatures);
					std::for_each(curvatures->begin(), curvatures->end(), [&pc1s, &pc2s](const auto &c) {
						if (!std::isnan(c.pc1)) pc1s.push_back(c.pc1);
						if (!std::isnan(c.pc2)) pc2s.push_back(c.pc2);
					});

					//auto threshold_it1 = pc1s.begin() + (num_curvatures - num_samples);
					auto threshold_it1 = pc1s.begin() + std::max((int)pc1s.size() - (int)num_samples, 0);
					std::nth_element(pc1s.begin(), threshold_it1, pc1s.end());
					std::cout << "\t#points to sample by pc1 = " << std::distance(threshold_it1, pc1s.end()) << std::endl;
					std::cout << "\tThreshold of pc1 = " << *threshold_it1 << std::endl;
					std::cout << "\t";
					std::copy(threshold_it1, threshold_it1 + 10, std::ostream_iterator<float>(std::cout, ", "));
					std::cout << std::endl;

					//auto threshold_it2 = pc2s.begin() + (num_curvatures - num_samples);
					auto threshold_it2 = pc2s.begin() + std::max((int)pc2s.size() - (int)num_samples, 0);
					std::nth_element(pc2s.begin(), threshold_it2, pc2s.end());
					std::cout << std::endl;
					std::cout << "\t#points to sample by pc1 = " << std::distance(threshold_it2, pc2s.end()) << std::endl;
					std::cout << "\tThreshold of pc2 = " << *threshold_it2 << std::endl;
					std::cout << "\t";
					std::copy(threshold_it2, threshold_it2 + 10, std::ostream_iterator<float>(std::cout, ", "));
					std::cout << std::endl;
				}
			}
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_pcl {

void sampling()
{
	//local::resampling_tutorial();

	local::subsampling_test();
}

}  // namespace my_pcl
