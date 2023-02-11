#include <cmath>
#include <memory>
#include <string>
#include <open3d/Open3D.h>


namespace {
namespace local {

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/OfflineSLAM.cpp
void offline_slam_example()
{
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Info);

	const open3d::data::SampleRedwoodRGBDImages sample_rgbd_data;
	const auto& color_filenames = sample_rgbd_data.GetColorPaths();
	const auto& depth_filenames = sample_rgbd_data.GetDepthPaths();

	const std::string device_code("CPU:0");
	const auto iterations = std::min<size_t>(color_filenames.size(), 100);
	const bool pointcloud = true;
	const bool visualize = true;

	const open3d::core::Device device(device_code);
	open3d::utility::LogInfo("Using device: {}", device.ToString());

	// Intrinsics.
	const open3d::camera::PinholeCameraIntrinsic intrinsic(open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
	const auto& focal_length = intrinsic.GetFocalLength();
	const auto& principal_point = intrinsic.GetPrincipalPoint();
	const open3d::core::Tensor intrinsic_t = open3d::core::Tensor::Init<double>({
		{focal_length.first, 0, principal_point.first},
		{0, focal_length.second, principal_point.second},
		{0, 0, 1}
	});

	// VoxelBlock configurations.
	const float voxel_size = 3.0f / 512.0f;
	const float trunc_voxel_multiplier = 8.0f;

	const int block_resolution = 16;
	const int block_count = 10000;

	// Odometry configurations.
	const float depth_scale = 1000.0f;
	const float depth_max = 3.0f;
	const float depth_diff = 0.07f;

	// Initialization.
	open3d::core::Tensor T_frame_to_model = open3d::core::Tensor::Eye(4, open3d::core::Dtype::Float64, device);

	open3d::t::pipelines::slam::Model model(voxel_size, block_resolution, block_count, T_frame_to_model, device);

	// Initialize frame.
	const auto ref_depth = open3d::t::io::CreateImageFromFile(depth_filenames[0]);
	open3d::t::pipelines::slam::Frame input_frame(int(ref_depth->GetRows()), int(ref_depth->GetCols()), intrinsic_t, device);
	open3d::t::pipelines::slam::Frame raycast_frame(int(ref_depth->GetRows()), int(ref_depth->GetCols()), intrinsic_t, device);

	// Iterate over frames.
	for (size_t i = 0; i < iterations; ++i)
	{
		open3d::utility::LogInfo("Processing {}/{}...", i, iterations);
		// Load image into frame.
		const auto input_depth = open3d::t::io::CreateImageFromFile(depth_filenames[i]);
		const auto input_color = open3d::t::io::CreateImageFromFile(color_filenames[i]);
		input_frame.SetDataFromImage("depth", *input_depth);
		input_frame.SetDataFromImage("color", *input_color);

		bool tracking_success = true;
		if (i > 0)
		{
			const auto& result = model.TrackFrameToModel(input_frame, raycast_frame, depth_scale, depth_max, depth_diff);

			const open3d::core::Tensor translation = result.transformation_.Slice(0, 0, 3).Slice(1, 3, 4);
			const double translation_norm = std::sqrt((translation * translation).Sum({0, 1}).Item<double>());

			// TODO(wei): more systematical failure check.
			// If the overlap is too small or translation is too high between
			// two consecutive frames, it is likely that the tracking failed.
			if (result.fitness_ >= 0.1 && translation_norm < 0.15)
			{
				T_frame_to_model = T_frame_to_model.Matmul(result.transformation_);
			}
			else  // Don't update.
			{
				tracking_success = false;
				open3d::utility::LogWarning(
					"Tracking failed for frame {}, fitness: {:.3f}, translation: {:.3f}. Using previous frame's pose.",
					i, result.fitness_, translation_norm
				);
			}
		}

		// Integrate.
		model.UpdateFramePose(int(i), T_frame_to_model);
		if (tracking_success)
		{
			model.Integrate(input_frame, depth_scale, depth_max, trunc_voxel_multiplier);
		}
		model.SynthesizeModelFrame(raycast_frame, depth_scale, 0.1f, depth_max, trunc_voxel_multiplier, false);
	}

	if (pointcloud)
	{
		const std::string filename("pcd_" + device.ToString() + ".ply");
		const auto& pcd = model.ExtractPointCloud();
		const auto pcd_legacy = std::make_shared<open3d::geometry::PointCloud>(pcd.ToLegacy());
		open3d::io::WritePointCloud(filename, *pcd_legacy);

		if (visualize)
		{
			open3d::visualization::Draw({pcd_legacy}, "Extracted PointCloud.");
		}
	}
	else
	{
		// If nothing is specified, draw and save the geometry as mesh.
		const std::string filename("mesh_" + device.ToString() + ".ply");
		const auto& mesh = model.ExtractTriangleMesh();
		const auto mesh_legacy = std::make_shared<open3d::geometry::TriangleMesh>(mesh.ToLegacy());
		open3d::io::WriteTriangleMesh(filename, *mesh_legacy);

		if (visualize)
		{
			open3d::visualization::Draw({mesh_legacy}, "Extracted Mesh.");
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

void slam()
{
	local::offline_slam_example();
}

}  // namespace my_open3d
