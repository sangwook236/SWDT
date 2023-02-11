#include <memory>
#include <tuple>
#include <string>
#include <Eigen/Core>
#include <open3d/Open3D.h>


namespace {
namespace local {

std::tuple<std::shared_ptr<open3d::geometry::PointCloud>, std::shared_ptr<open3d::geometry::PointCloud>, std::shared_ptr<open3d::pipelines::registration::Feature>>
//PreprocessPointCloud(const char *file_name, const double voxel_size)
PreprocessPointCloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, const double voxel_size)
{
	//auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
	const auto pcd_down = pcd->VoxelDownSample(voxel_size);
	pcd_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
	const auto pcd_fpfh = open3d::pipelines::registration::ComputeFPFHFeature(
		*pcd_down,
		open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100)
	);
	return std::make_tuple(pcd, pcd_down, pcd_fpfh);
}

void VisualizeRegistration(const open3d::geometry::PointCloud &source, const open3d::geometry::PointCloud &target, const Eigen::Matrix4d &transformation)
{
	auto source_transformed_ptr(std::make_shared<open3d::geometry::PointCloud>());
	auto target_ptr(std::make_shared<open3d::geometry::PointCloud>());
	*source_transformed_ptr = source;
	*target_ptr = target;
	source_transformed_ptr->Transform(transformation);
	open3d::visualization::DrawGeometries({source_transformed_ptr, target_ptr}, "Registration result");
}

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/RegistrationColoredICP.cpp
void color_icp_example()
{
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);

	const double depth_scale = 1000.0, depth_trunc = 3.0;
	const bool convert_rgb_to_intensity = true;

	const bool visualize = true;

	//-----
	const open3d::data::SampleRedwoodRGBDImages sample_rgbd_data;
	const auto& color_filenames = sample_rgbd_data.GetColorPaths();
	const auto& depth_filenames = sample_rgbd_data.GetDepthPaths();

	// Intrinsics.
	const open3d::camera::PinholeCameraIntrinsic intrinsic(open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

	open3d::geometry::Image color, depth;
	open3d::io::ReadImage(color_filenames[0], color);
	open3d::io::ReadImage(depth_filenames[0], depth);
	const auto source_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto source = open3d::geometry::PointCloud::CreateFromRGBDImage(*source_rgbd, intrinsic);

	open3d::io::ReadImage(color_filenames[1], color);
	open3d::io::ReadImage(depth_filenames[1], depth);
	const auto target_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto target = open3d::geometry::PointCloud::CreateFromRGBDImage(*target_rgbd, intrinsic);

	//-----
	const std::vector<double> voxel_sizes({0.05, 0.05 / 2, 0.05 / 4});
	const std::vector<int> iterations({50, 30, 14});
	Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
	for (int i = 0; i < 3; ++i)
	{
		const double& voxel_size = voxel_sizes[i];

		auto source_down = source->VoxelDownSample(voxel_size);
		source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0, 30));

		auto target_down = target->VoxelDownSample(voxel_size);
		target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0, 30));

		const auto& result = open3d::pipelines::registration::RegistrationColoredICP(
			*source_down, *target_down,
			0.07, trans,
			open3d::pipelines::registration::TransformationEstimationForColoredICP(),
			open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iterations[i])
		);
		trans = result.transformation_;

		if (visualize)
		{
			VisualizeRegistration(*source, *target, trans);
		}
	}

	std::cout << "Final transformation:\n{}." << trans << std::endl;
}

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/RegistrationRANSAC.cpp
void ransac_registration_example()
{
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);

	const std::string method("feature_matching");
	const std::string kMethodFeature("feature_matching");
	const std::string kMethodCorres("correspondence");

	const double depth_scale = 1000.0, depth_trunc = 3.0;
	const bool convert_rgb_to_intensity = true;

	const bool mutual_filter = true;
	const double voxel_size = 0.05;
	const double distance_multiplier = 1.5;
	const double distance_threshold = voxel_size * distance_multiplier;
	const int max_iterations = 1000000;
	const double confidence = 0.999;

	//-----
	const open3d::data::SampleRedwoodRGBDImages sample_rgbd_data;
	const auto& color_filenames = sample_rgbd_data.GetColorPaths();
	const auto& depth_filenames = sample_rgbd_data.GetDepthPaths();

	// Intrinsics.
	const open3d::camera::PinholeCameraIntrinsic intrinsic(open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

	open3d::geometry::Image color, depth;
	open3d::io::ReadImage(color_filenames[0], color);
	open3d::io::ReadImage(depth_filenames[0], depth);
	const auto source_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto source_pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*source_rgbd, intrinsic);

	open3d::io::ReadImage(color_filenames[1], color);
	open3d::io::ReadImage(depth_filenames[1], depth);
	const auto target_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto target_pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*target_rgbd, intrinsic);

	// Prepare input.
	std::shared_ptr<open3d::geometry::PointCloud> source, source_down, target, target_down;
	std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh, target_fpfh;
	std::tie(source, source_down, source_fpfh) = PreprocessPointCloud(source_pcd, voxel_size);
	std::tie(target, target_down, target_fpfh) = PreprocessPointCloud(target_pcd, voxel_size);

	//-----
	// Prepare checkers.
	std::vector<std::reference_wrapper<const open3d::pipelines::registration::CorrespondenceChecker>> correspondence_checker;
	const auto correspondence_checker_edge_length = open3d::pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
	const auto correspondence_checker_distance = open3d::pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
	correspondence_checker.push_back(correspondence_checker_edge_length);
	correspondence_checker.push_back(correspondence_checker_distance);

	open3d::pipelines::registration::RegistrationResult registration_result;
	if (method == kMethodFeature)
	{
		registration_result = open3d::pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
			*source_down, *target_down,
			*source_fpfh, *target_fpfh,
			mutual_filter, distance_threshold,
			open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
			3, correspondence_checker,
			open3d::pipelines::registration::RANSACConvergenceCriteria(max_iterations, confidence)
		);
	}
	else if (method == kMethodCorres)
	{
		// Manually search correspondences.
		const int nPti = int(source_down->points_.size());
		const int nPtj = int(target_down->points_.size());

		open3d::geometry::KDTreeFlann feature_tree_i(*source_fpfh);
		open3d::geometry::KDTreeFlann feature_tree_j(*target_fpfh);

		open3d::pipelines::registration::CorrespondenceSet corres_ji;
		std::vector<int> i_to_j(nPti, -1);

		// Buffer all correspondences.
		for (int j = 0; j < nPtj; ++j)
		{
			std::vector<int> corres_tmp(1);
			std::vector<double> dist_tmp(1);

			feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)), 1, corres_tmp, dist_tmp);
			int i = corres_tmp[0];
			corres_ji.push_back(Eigen::Vector2i(i, j));
		}

		if (mutual_filter)
		{
			open3d::pipelines::registration::CorrespondenceSet mutual_corres;
			for (auto &corres : corres_ji)
			{
				int j = corres(1);
				int j2i = corres(0);

				std::vector<int> corres_tmp(1);
				std::vector<double> dist_tmp(1);
				feature_tree_j.SearchKNN(Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1, corres_tmp, dist_tmp);
				int i2j = corres_tmp[0];
				if (i2j == j)
				{
					mutual_corres.push_back(corres);
				}
			}

			open3d::utility::LogDebug("{:d} points remain after mutual filter", mutual_corres.size());
			registration_result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
				*source_down, *target_down,
				mutual_corres,
				distance_threshold,
				open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
				3, correspondence_checker,
				open3d::pipelines::registration::RANSACConvergenceCriteria(max_iterations, confidence)
			);
		}
		else
		{
			open3d::utility::LogDebug("{:d} points remain", corres_ji.size());
			registration_result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
				*source_down, *target_down,
				corres_ji,
				distance_threshold,
				open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
				3, correspondence_checker,
				open3d::pipelines::registration::RANSACConvergenceCriteria(max_iterations, confidence)
			);
		}
	}

	VisualizeRegistration(*source, *target, registration_result.transformation_);
}

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/RegistrationFGR.cpp
void fgr_registration_example()
{
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);

	const double depth_scale = 1000.0, depth_trunc = 3.0;
	const bool convert_rgb_to_intensity = true;

	const double voxel_size = 0.05;
	const double distance_multiplier = 1.5;
	const double distance_threshold = voxel_size * distance_multiplier;
	const int max_iterations = 64;
	const int max_tuples = 1000;

	//-----
	const open3d::data::SampleRedwoodRGBDImages sample_rgbd_data;
	const auto& color_filenames = sample_rgbd_data.GetColorPaths();
	const auto& depth_filenames = sample_rgbd_data.GetDepthPaths();

	// Intrinsics.
	const open3d::camera::PinholeCameraIntrinsic intrinsic(open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

	open3d::geometry::Image color, depth;
	open3d::io::ReadImage(color_filenames[0], color);
	open3d::io::ReadImage(depth_filenames[0], depth);
	const auto source_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto source_pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*source_rgbd, intrinsic);

	open3d::io::ReadImage(color_filenames[1], color);
	open3d::io::ReadImage(depth_filenames[1], depth);
	const auto target_rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	const auto target_pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*target_rgbd, intrinsic);

	// Prepare input.
	std::shared_ptr<open3d::geometry::PointCloud> source, source_down, target, target_down;
	std::shared_ptr<open3d::pipelines::registration::Feature> source_fpfh, target_fpfh;
	std::tie(source, source_down, source_fpfh) = PreprocessPointCloud(source_pcd, voxel_size);
	std::tie(target, target_down, target_fpfh) = PreprocessPointCloud(target_pcd, voxel_size);

	//-----
	const auto &registration_result = open3d::pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(
		*source_down, *target_down,
		*source_fpfh, *target_fpfh,
		open3d::pipelines::registration::FastGlobalRegistrationOption(
			/*division_factor =*/ 1.4,
			/*use_absolute_scale =*/ true,
			/*decrease_mu =*/ true, 
			/*maximum_correspondence_distance =*/ distance_threshold,
			/*iteration_number =*/ max_iterations,
			/*tuple_scale =*/ 0.95,
			/*maximum_tuple_count =*/ max_tuples,
			/*tuple_test =*/ true
		)
	);

	VisualizeRegistration(*source, *target, registration_result.transformation_);
}

std::vector<std::shared_ptr<open3d::geometry::PointCloud>> load_point_clouds(const double voxel_size = 0.0)
{
	open3d::data::DemoICPPointClouds demo_icp_pcds;
	const auto& paths = demo_icp_pcds.GetPaths();
	std::vector<std::shared_ptr<open3d::geometry::PointCloud>> pcds_down;
	for (auto it = paths.begin(); it != paths.end(); ++it)
	{
		open3d::geometry::PointCloud pcd;
		open3d::io::ReadPointCloud(*it, pcd, open3d::io::ReadPointCloudOption());
		const auto pcd_down = pcd.VoxelDownSample(voxel_size);
		pcds_down.push_back(pcd_down);
	}

	return pcds_down;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix6d> pairwise_registration(const std::shared_ptr<open3d::geometry::PointCloud>& source, const std::shared_ptr<open3d::geometry::PointCloud>& target, const double max_correspondence_distance_coarse, const double max_correspondence_distance_fine)
{
	std::cout << "Apply point-to-plane ICP." << std::endl;
	const auto& icp_coarse = open3d::pipelines::registration::RegistrationICP(
		*source, *target, max_correspondence_distance_coarse,
		Eigen::Matrix4d::Identity(),
		open3d::pipelines::registration::TransformationEstimationPointToPlane()
	);
	const auto& icp_fine = open3d::pipelines::registration::RegistrationICP(
		*source, *target, max_correspondence_distance_fine,
		icp_coarse.transformation_,
		open3d::pipelines::registration::TransformationEstimationPointToPlane()
	);
	const auto& information_icp = open3d::pipelines::registration::GetInformationMatrixFromPointClouds(
		*source, *target, max_correspondence_distance_fine,
		icp_fine.transformation_
	);
	return std::make_pair(icp_fine.transformation_, information_icp);
}

open3d::pipelines::registration::PoseGraph full_registration(const std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& pcds, const double max_correspondence_distance_coarse, const double max_correspondence_distance_fine)
{
	open3d::pipelines::registration::PoseGraph pose_graph;
	Eigen::Matrix4d odometry(Eigen::Matrix4d::Identity());
	pose_graph.nodes_.push_back(open3d::pipelines::registration::PoseGraphNode(odometry));
	for (size_t source_id = 0; source_id < pcds.size(); ++source_id)
	{
		for (size_t target_id = source_id + 1; target_id < pcds.size(); ++target_id)
		{
			const auto& result = pairwise_registration(pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine);
			const auto& transformation_icp = result.first;
			const auto& information_icp = result.second;

			std::cout << "Build open3d::pipelines::registration::PoseGraph." << std::endl;
			if (source_id + 1 == target_id)  // Odometry case.
			{
				odometry = transformation_icp * odometry;
				pose_graph.nodes_.push_back(open3d::pipelines::registration::PoseGraphNode(odometry.inverse()));
				pose_graph.edges_.push_back(open3d::pipelines::registration::PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, false));
			}
			else  // Loop closure case.
			{
				pose_graph.edges_.push_back(open3d::pipelines::registration::PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, true));
			}
		}
	}
	return pose_graph;
}

// REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html
void multiway_registration_tutorial()
{
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);

	const double voxel_size = 0.02;
	const auto& pcds_down = load_point_clouds(voxel_size);

	std::cout << "#point clouds = " << pcds_down.size() << std::endl;
	std::cout << "#points = " << std::accumulate(pcds_down.begin(), pcds_down.end(), 0, [](const size_t count, const std::shared_ptr<open3d::geometry::PointCloud> pcd) { return count + pcd->points_.size(); }) << std::endl;

	Eigen::Vector3d lookat({2.6172, 2.0475, 1.532});
	Eigen::Vector3d up({-0.0694, -0.9768, 0.2024});
	Eigen::Vector3d front({0.4257, -0.2125, -0.8795});
	double zoom = 0.3412;
	open3d::visualization::DrawGeometries(
		std::vector<std::shared_ptr<const open3d::geometry::Geometry>>(pcds_down.begin(), pcds_down.end()),
		"Registration result",
		/*width =*/ 640, /*height =*/ 480, /*left =*/ 50, /*top =*/ 50,
		/*point_show_normal =*/ false, /*mesh_show_wireframe =*/ false, /*mesh_show_back_face =*/ false,
		/*lookat =*/ &lookat, /*up =*/ &up, /*front =*/ &front,
		/*zoom =*/ &zoom
	);

	//-----
	std::cout << "Full registration ..." << std::endl;
	const double max_correspondence_distance_coarse = voxel_size * 15.0;
	const double max_correspondence_distance_fine = voxel_size * 1.5;

	auto pose_graph = full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine);

	//-----
	std::cout << "Optimizing PoseGraph ..." << std::endl;
	const open3d::pipelines::registration::GlobalOptimizationOption option(
		/*max_correspondence_distance =*/ max_correspondence_distance_fine,
		/*edge_prune_threshold =*/ 0.25,
		/*preference_loop_closure =*/ 1.0,
		/*reference_node =*/ 0
	);

	open3d::pipelines::registration::GlobalOptimization(
		pose_graph,
		open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
		open3d::pipelines::registration::GlobalOptimizationConvergenceCriteria(),
		option
	);

	//-----
	// Visualize optimization.
	std::cout << "Transform points and display." << std::endl;
	for (size_t point_id = 0; point_id < pcds_down.size(); ++point_id)
	{
		std::cout << pose_graph.nodes_[point_id].pose_ << std::endl;
		pcds_down[point_id]->Transform(pose_graph.nodes_[point_id].pose_);
	}

	open3d::visualization::DrawGeometries(
		std::vector<std::shared_ptr<const open3d::geometry::Geometry>>(pcds_down.begin(), pcds_down.end()),
		"Registration optimized",
		/*width =*/ 640, /*height =*/ 480, /*left =*/ 50, /*top =*/ 50,
		/*point_show_normal =*/ false, /*mesh_show_wireframe =*/ false, /*mesh_show_back_face =*/ false,
		/*lookat =*/ &lookat, /*up =*/ &up, /*front =*/ &front,
		/*zoom =*/ &zoom
	);

	//-----
	// Make a combined point cloud.
	const auto& pcds = load_point_clouds(voxel_size);
	open3d::geometry::PointCloud pcd_combined;
	for (size_t point_id = 0; point_id < pcds.size(); ++point_id)
	{
		pcds[point_id]->Transform(pose_graph.nodes_[point_id].pose_);
		pcd_combined += *pcds[point_id];
	}
	const auto pcd_combined_down = pcd_combined.VoxelDownSample(voxel_size);
	open3d::io::WritePointCloud("./multiway_registration.pcd", *pcd_combined_down);

	open3d::visualization::DrawGeometries(
		{pcd_combined_down},
		"Combined point cloud",
		/*width =*/ 640, /*height =*/ 480, /*left =*/ 50, /*top =*/ 50,
		/*point_show_normal =*/ false, /*mesh_show_wireframe =*/ false, /*mesh_show_back_face =*/ false,
		/*lookat =*/ &lookat, /*up =*/ &up, /*front =*/ &front,
		/*zoom =*/ &zoom
	);
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

void registration()
{
	//local::color_icp_example();

	//local::ransac_registration_example();
	local::fgr_registration_example();

	//local::multiway_registration_tutorial();
}

}  // namespace my_open3d
