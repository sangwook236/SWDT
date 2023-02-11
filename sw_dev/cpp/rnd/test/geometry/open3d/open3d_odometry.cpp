#include <memory>
#include <tuple>
#include <string>
#include <Eigen/Core>
#include <open3d/Open3D.h>


namespace {
namespace local {

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/RGBDOdometry.cpp
void rgbd_odometry_example()
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
	const auto source = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	if (visualize)
	{
		const auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*source, intrinsic);
		open3d::visualization::DrawGeometries({pcd}, "RGBD Image #1");
	}

	open3d::io::ReadImage(color_filenames[1], color);
	open3d::io::ReadImage(depth_filenames[1], depth);
	const auto target = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity);
	if (visualize)
	{
		const auto pcd = open3d::geometry::PointCloud::CreateFromRGBDImage(*target, intrinsic);
		open3d::visualization::DrawGeometries({pcd}, "RGBD Image #2");
	}

	const auto odo_init(Eigen::Matrix4d::Identity());
	const auto& rgbd_odo = open3d::pipelines::odometry::ComputeRGBDOdometry(
		*source, *target,
		intrinsic,
		odo_init,
		open3d::pipelines::odometry::RGBDOdometryJacobianFromHybridTerm(),
		open3d::pipelines::odometry::OdometryOption()
	);

	if (std::get<0>(rgbd_odo))
	{
		std::cout << "RGBD Odometry:" << std::endl;
		std::cout << std::get<1>(rgbd_odo) << std::endl;

		std::cout << "Information matrix:" << std::endl;
		std::cout << std::get<2>(rgbd_odo) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

void odometry()
{
	local::rgbd_odometry_example();
}

}  // namespace my_open3d
