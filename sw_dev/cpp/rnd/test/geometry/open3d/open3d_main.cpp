#include <memory>
#include <chrono>
#include <string>
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

void io_example()
{
	const std::string point_cloud_filepath("/path/to/sample.pcd");
	//const std::string point_cloud_filepath("/path/to/sample.ply");

	auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
	if (open3d::io::ReadPointCloud(point_cloud_filepath, *pointCloud))
		std::cout << "A point cloud loaded from " << point_cloud_filepath << std::endl;
	else
	{
		std::cerr << "Failed to load a point cloud from " << point_cloud_filepath << std::endl;
		return;
	}

	if (!pointCloud->HasNormals())
		pointCloud->EstimateNormals();
	pointCloud->NormalizeNormals();
	pointCloud->PaintUniformColor({0.5, 0.5, 0.5});

	//pointCloud = pointCloud->VoxelDownSample(10.0);
	std::cout << "#points loaded = " << pointCloud->points_.size() << std::endl;

	open3d::visualization::DrawGeometries({pointCloud}, "Point Cloud", 1000, 800);
}

std::string indent(const int n)
{
	const std::string s("    ");
	std::string cs;
	for (int i = 0; i < n; ++i)
		cs += s;
	return cs;
}

// REF [site] >> http://www.open3d.org/docs/release/tutorial/geometry/octree.html
bool traverse_octree_node(const std::shared_ptr<open3d::geometry::OctreeNode> &node, const std::shared_ptr<open3d::geometry::OctreeNodeInfo> &nodeInfo)
{
	bool early_stop = false;

	if (open3d::geometry::OctreeInternalNode *inode = dynamic_cast<open3d::geometry::OctreeInternalNode *>(node.get()))
	{
		if (open3d::geometry::OctreeInternalPointNode *ipnode = dynamic_cast<open3d::geometry::OctreeInternalPointNode*>(node.get()))
		{
			int n = 0;
			for (const auto &child: ipnode->children_)
				if (child != nullptr)
					n += 1;
			std::cout << indent(nodeInfo->depth_) << nodeInfo->child_index_ << ": Internal node at depth " << nodeInfo->depth_ << " has " << n << " children and " << ipnode->indices_.size() << " points (" << nodeInfo->origin_.transpose() << ")" << std::endl;

			// We only want to process nodes / spatial regions with enough points.
			early_stop = ipnode->indices_.size() < 250;
		}
	}
	else if (open3d::geometry::OctreeLeafNode *lnode = dynamic_cast<open3d::geometry::OctreeLeafNode *>(node.get()))
	{
		//if (open3d::geometry::OctreeColorLeafNode *plnode = dynamic_cast<open3d::geometry::OctreeColorLeafNode *>(node.get()))
		if (open3d::geometry::OctreePointColorLeafNode *plnode = dynamic_cast<open3d::geometry::OctreePointColorLeafNode *>(node.get()))
			std::cout << indent(nodeInfo->depth_) << nodeInfo->child_index_ << ": Leaf node at depth " << nodeInfo->depth_ << " has " << plnode->indices_.size() << " points (" << nodeInfo->origin_.transpose() << ")" << std::endl;
	}
	else
		throw std::runtime_error("Node type not recognized!");

	// Early stopping: if true, traversal of children of the current node will be skipped.
	return early_stop;
}

void octree_example()
{
	const std::string point_cloud_filepath("/path/to/sample.pcd");
	//const std::string point_cloud_filepath("/path/to/sample.ply");

	auto pointCloud = std::make_shared<open3d::geometry::PointCloud>();
	if (open3d::io::ReadPointCloud(point_cloud_filepath, *pointCloud))
		std::cout << "A point cloud loaded from " << point_cloud_filepath << std::endl;
	else
	{
		std::cerr << "Failed to load a point cloud from " << point_cloud_filepath << std::endl;
		return;
	}

	//pointCloud = pointCloud->VoxelDownSample(10.0);
	std::cout << "#points = " << pointCloud->points_.size() << std::endl;

	//--------------------
	const size_t octree_max_depth = 4;
#if 0
	const Eigen::Vector3d octree_origin(-121.5, -221.5, 0.0);
	const double octree_size = 1052.42;
#else
	const Eigen::Vector3d octree_origin(0.0, 0.0, 0.0);
	const double octree_size = 1500.0;
#endif
	const Eigen::Vector3d color(0.5, 0.5, 0.5);

	const auto start_time(std::chrono::high_resolution_clock::now());
#if 1
	open3d::geometry::Octree octree(octree_max_depth);
	octree.ConvertFromPointCloud(*pointCloud);
#elif 0
	open3d::geometry::Octree octree(octree_max_depth, octree_origin, octree_size);
	std::for_each(pointCloud->points_.begin(), pointCloud->points_.end(), [&octree, &color](const auto &point) { octree.InsertPoint(point, open3d::geometry::OctreeColorLeafNode::GetInitFunction(), open3d::geometry::OctreeColorLeafNode::GetUpdateFunction(color)); });
#else
	open3d::geometry::Octree octree(octree_max_depth, octree_origin, octree_size);
	int pidx = 0;
	std::for_each(pointCloud->points_.begin(), pointCloud->points_.end(), [&octree, &color, &pidx](const auto &point) { octree.InsertPoint(point, open3d::geometry::OctreePointColorLeafNode::GetInitFunction(), open3d::geometry::OctreePointColorLeafNode::GetUpdateFunction(pidx, color), open3d::geometry::OctreeInternalPointNode::GetInitFunction(), open3d::geometry::OctreeInternalPointNode::GetUpdateFunction(pidx)); ++pidx; });
#endif
	const auto elapsed_time(std::chrono::high_resolution_clock::now() - start_time);
	std::cout << "Point cloud added to an octree: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count() << " ,secs." << std::endl;

	std::cout << "\tMax depth = " << octree.max_depth_ << std::endl;
	std::cout << "\tOrigin = " << octree.origin_.transpose() << std::endl;
	std::cout << "\tSize = " << octree.size_ << std::endl;
	//std::cout << "\tRoot node = " << octree.root_node_ << std::endl;
	if (open3d::geometry::OctreeInternalPointNode *root_node = dynamic_cast<open3d::geometry::OctreeInternalPointNode*>(octree.root_node_.get()))
		std::cout << "\t#child nodes = " << root_node->indices_.size() << std::endl;

	octree.Traverse(traverse_octree_node);
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

}  // namespace my_open3d

int open3d_main(int argc, char *argv[])
{
	//open3d::utility::LogInfo("Info log.");

	local::draw_example();
	//local::io_example();
	//local::octree_example();

	return 0;
}
