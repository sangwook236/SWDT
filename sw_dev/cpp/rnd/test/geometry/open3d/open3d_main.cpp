#include <iostream>
#include <open3d/Open3D.h>


namespace {
namespace local {

// REF [site] >> https://github.com/isl-org/open3d-cmake-find-package/blob/master/Draw.cpp
void draw_example()
{
	const auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0);
	sphere->ComputeVertexNormals();
	sphere->PaintUniformColor({0.0, 1.0, 0.0});
	open3d::visualization::DrawGeometries({sphere});
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

}  // namespace my_open3d

int open3d_main(int argc, char *argv[])
{
	//open3d::utility::LogInfo("Info log.");

	local::draw_example();

	return 0;
}
