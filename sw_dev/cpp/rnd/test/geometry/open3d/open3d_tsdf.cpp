#include <vector>
#include <string>
#include <iostream>
#include <open3d/Open3D.h>


namespace {
namespace local {

// REF [site] >>
//  https://github.com/isl-org/Open3D/blob/main/examples/cpp/IntegrateRGBD.cpp
//  https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html
void integrate_rgbd_example()
{
	const bool save_pointcloud = false;
	const bool save_mesh = false;
	const bool save_voxel = false;
	const int every_k_frames = 0;
	const double length = 4.0;
	const int resolution = 512;
	const double sdf_trunc_percentage = 0.01;
	const int verbose = 5;
	open3d::utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)verbose);

	// Download data

	// Redwood RGBD:
	//	Color: 640x480, RGB, 24 bits
	//	Depth: 640x480, 16 bits

	//	${USERPROFILE}/open3d_data/download/SampleRedwoodRGBDImages
	//	${USERPROFILE}/open3d_data/extract/SampleRedwoodRGBDImages
	//open3d::data::SampleRedwoodRGBDImages data(/*data_root =*/);
	//	./download/SampleRedwoodRGBDImages
	//	./extract/SampleRedwoodRGBDImages
	open3d::data::SampleRedwoodRGBDImages data(/*data_root =*/ "./");

	// Read trajectory

	const auto camera_trajectory = open3d::io::CreatePinholeCameraTrajectoryFromFile(data.GetTrajectoryLogPath());
	const std::string dir_name(open3d::utility::filesystem::GetFileParentDirectory(data.GetRGBDMatchPath()));
	FILE *file = open3d::utility::filesystem::FOpen(data.GetRGBDMatchPath(), "r");
	if (file == nullptr)
	{
		open3d::utility::LogWarning("Unable to open file {}", data.GetRGBDMatchPath());
		return;
	}

	// TSDF volume integration

	open3d::pipelines::integration::ScalableTSDFVolume volume(
		/*voxel_length =*/ length / double(resolution),
		/*sdf_trunc =*/ length * sdf_trunc_percentage,
		/*color_type =*/ open3d::pipelines::integration::TSDFVolumeColorType::RGB8
	);

	open3d::utility::FPSTimer timer("Process RGBD stream", (int)camera_trajectory->parameters_.size());
	char buffer[DEFAULT_IO_BUFFER_SIZE];
	size_t index = 0;
	size_t save_index = 0;
	open3d::geometry::Image depth, color;
	while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file))
	{
		const std::vector<std::string> &st = open3d::utility::SplitString(buffer, "\t\r\n ");
		if (st.size() >= 2)
		{
			open3d::utility::LogInfo("Processing frame {:d} ...", index);
			if (!open3d::io::ReadImage(dir_name + st[0], depth))
			{
				std::cerr << "Depth image file not found, " << dir_name + st[0] << std::endl;
			}
			if (!open3d::io::ReadImage(dir_name + st[1], color))
			{
				std::cerr << "RGB image file not found, " << dir_name + st[1] << std::endl;
			}
			const auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
				color, depth,
				/*depth_scale =*/ 1000.0,
				/*depth_trunc =*/ 4.0,
				/*convert_rgb_to_intensity =*/ false
			);
			if (index == 0 || (every_k_frames > 0 && index % every_k_frames == 0))
				volume.Reset();
			volume.Integrate(
				*rgbd,
				/*intrinsic =*/ camera_trajectory->parameters_[index].intrinsic_,
				/*extrinsic =*/ camera_trajectory->parameters_[index].extrinsic_
			);

			++index;
			if (index == camera_trajectory->parameters_.size() || (every_k_frames > 0 && index % every_k_frames == 0))
			{
				open3d::utility::LogInfo("Saving fragment {:d} ...", save_index);
				std::string save_index_str = std::to_string(save_index);
				if (save_pointcloud)
				{
					open3d::utility::LogInfo("Saving pointcloud {:d} ...", save_index);
					const auto pcd = volume.ExtractPointCloud();
					open3d::io::WritePointCloud("pointcloud_" + save_index_str + ".ply", *pcd);
				}
				if (save_mesh)
				{
					open3d::utility::LogInfo("Saving mesh {:d} ...", save_index);
					const auto mesh = volume.ExtractTriangleMesh();
					open3d::io::WriteTriangleMesh("mesh_" + save_index_str + ".ply", *mesh);
				}
				if (save_voxel)
				{
					open3d::utility::LogInfo("Saving voxel {:d} ...", save_index);
					const auto voxel = volume.ExtractVoxelPointCloud();
					open3d::io::WritePointCloud("voxel_" + save_index_str + ".ply", *voxel);
				}
				++save_index;
			}

			timer.Signal();
		}
	}
	fclose(file);

	// Extract a mesh

	std::cout << "Extract a triangle mesh from the volume and visualize it." << std::endl;
	auto mesh = volume.ExtractTriangleMesh();
	mesh->ComputeVertexNormals();

	Eigen::Vector3d lookat{2.0712, 2.0312, 1.7251};
	Eigen::Vector3d up{-0.0558, -0.9809, 0.1864};
	Eigen::Vector3d front{0.5297, -0.1873, -0.8272};
	double zoom = 0.47;
	open3d::visualization::DrawGeometries(
		{mesh},
		/*window_name =*/ "Open3D",
		/*width =*/ 1280,
		/*height =*/ 960,
		/*left =*/ 50,
		/*top =*/ 50,
		/*point_show_normal =*/ false,
		/*mesh_show_wireframe =*/ false,
		/*mesh_show_back_face =*/ false,
		/*lookat =*/ &lookat,
		/*up =*/ &up,
		/*front =*/ &front,
		/*zoom =*/ &zoom
	);
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

void tsdf()
{
	local::integrate_rgbd_example();
}

}  // namespace my_open3d
