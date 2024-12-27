#include <memory>
#include <chrono>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <open3d/Open3D.h>


namespace {
namespace local {

// REF [site] >> https://github.com/isl-org/open3d-cmake-find-package/blob/master/Draw.cpp
void draw_example()
{
	const auto sphere = open3d::geometry::TriangleMesh::CreateSphere(1.0);
	sphere->ComputeVertexNormals();
	sphere->PaintUniformColor({0.0, 1.0, 0.0});

	const auto box = open3d::geometry::TriangleMesh::CreateBox(1.0, 1.0, 1.0);
	box->ComputeVertexNormals();
	box->PaintUniformColor({0.0, 0.0, 1.0});
	box->Translate({2.0, 0.0, 0.0});

	// Visualize
#if 0
	open3d::visualization::DrawGeometries({sphere, box});
#elif 1
	open3d::visualization::Draw({sphere, box});  // Open3D Viewer
#else
	// REF [site] >> https://github.com/isl-org/Open3D/issues/2850

	open3d::visualization::Visualizer visualizer;
	visualizer.CreateVisualizerWindow("Open3D Viewer");

	visualizer.AddGeometry(sphere);
	visualizer.AddGeometry(box);

	//auto param = open3d::camera::PinholeCameraParameters();
	//open3d::io::ReadIJsonConvertible("./view_point.json", param);
	//auto view_control = visualizer.GetViewControl();
	//view_control.ConvertFromPinholeCameraParameters(param, true);

	visualizer.PollEvents();
	visualizer.UpdateRender();
	visualizer.Run();
	visualizer.Destroy();
#endif
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

	open3d::visualization::DrawGeometries({pointCloud}, "Open3D Viewer", 1000, 800);
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
	const size_t max_points = 250;
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
			if (max_points > 0) early_stop = ipnode->indices_.size() < max_points;
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
	const double size_expand = 0.01;
#if 1
	const Eigen::Array3d &min_bound = pointCloud->GetMinBound();
	const Eigen::Array3d &max_bound = pointCloud->GetMaxBound();
	const Eigen::Array3d center((min_bound + max_bound) / 2);
	//const Eigen::Array3d &center = pointCloud->GetCenter();  // Mean of the points.
	const Eigen::Array3d half_sizes(center - min_bound);
	const double max_half_size = half_sizes.maxCoeff();

	const Eigen::Vector3d octree_origin = min_bound.min(center - max_half_size);
	const double octree_size = max_half_size == 0 ? size_expand : (max_half_size * 2 * (1 + size_expand));
#else
	const Eigen::Vector3d octree_origin(-121.5, -221.5, 0.0);
	const double octree_size = 1052.42;
#endif
	const Eigen::Vector3d color(0.5, 0.5, 0.5);

	const auto start_time(std::chrono::high_resolution_clock::now());
#if 1
	open3d::geometry::Octree octree(octree_max_depth);
	octree.ConvertFromPointCloud(*pointCloud, size_expand);
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

// REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/cpp/PoseGraph.cpp
void pose_graph_example()
{
	const std::string input_pose_graph_filepath("../pose_graph_noisy.json");
	const std::string output_pose_graph_filepath("../pose_graph_optimized.json");

#if 1
	open3d::pipelines::registration::PoseGraph pose_graph_input;
	pose_graph_input.nodes_.push_back(open3d::pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Random()));
	pose_graph_input.nodes_.push_back(open3d::pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Random()));
	pose_graph_input.nodes_.push_back(open3d::pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Random()));
	pose_graph_input.edges_.push_back(open3d::pipelines::registration::PoseGraphEdge(0, 1, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false, 1.0));
	pose_graph_input.edges_.push_back(open3d::pipelines::registration::PoseGraphEdge(1, 2, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false, 1.0));
	pose_graph_input.edges_.push_back(open3d::pipelines::registration::PoseGraphEdge(0, 2, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), true, 0.2));
#elif 0
	open3d::pipelines::registration::PoseGraph pose_graph_input;
	if (!open3d::io::ReadPoseGraph(input_pose_graph_filepath, pose_graph_input))
	{
		std::cerr << "Failed to read a pose graph from " << input_pose_graph_filepath << std::endl;
		return;
	}
#else
	const auto pose_graph_input = *open3d::io::CreatePoseGraphFromFile(input_pose_graph_filepath);
#endif

	open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt optimization_method;
	open3d::pipelines::registration::GlobalOptimizationConvergenceCriteria criteria;
	open3d::pipelines::registration::GlobalOptimizationOption option;

	open3d::pipelines::registration::GlobalOptimization(pose_graph_input, optimization_method, criteria, option);
	const auto pose_graph_input_prunned = open3d::pipelines::registration::CreatePoseGraphWithoutInvalidEdges(pose_graph_input, option);

	if (!open3d::io::WritePoseGraph(output_pose_graph_filepath, *pose_graph_input_prunned))
	{
		std::cerr << "Failed to write a pose graph to " << output_pose_graph_filepath << std::endl;
		return;
	}
}

void tensor_test()
{
	//open3d::core::Device device("CPU");  // Runtime error
	//open3d::core::Device device("CPU:0");
	//open3d::core::Device device("CUDA");  // Runtime error
	open3d::core::Device device("CUDA:0");
	open3d::core::Dtype dtype(open3d::core::Float32);
	//open3d::core::Dtype dtype(open3d::core::Float64);

	{
		auto cloud_cpu(open3d::t::geometry::PointCloud(open3d::core::Device("CPU:0")));
		auto cloud_gpu_from_cpu(cloud_cpu.To(open3d::core::Device("CUDA:0")));
		auto cloud_gpu(open3d::t::geometry::PointCloud(open3d::core::Device("CUDA:0")));
		auto cloud_cpu_from_gpu(cloud_gpu.To(open3d::core::Device("CPU:0")));

		auto cloud_legacy_from_cpu(cloud_cpu.ToLegacy());
		auto cloud_legacy_from_gpu(cloud_gpu.ToLegacy());

		open3d::geometry::PointCloud cloud_legacy;
		const size_t num_points = 100;
		cloud_legacy.points_.reserve(num_points);
		cloud_legacy.colors_.reserve(num_points);
		for (size_t i = 0; i < num_points; ++i)
		{
			cloud_legacy.points_.push_back(Eigen::Vector3d::Random() * 100.0);
			cloud_legacy.colors_.push_back((Eigen::Vector3d::Random() + Eigen::Vector3d::Ones()) * 0.5);
		}
		auto cloud_cpu_from_legacy(open3d::t::geometry::PointCloud::FromLegacy(cloud_legacy, open3d::core::Float32, open3d::core::Device("CPU:0")));
		//auto cloud_gpu_from_legacy(open3d::t::geometry::PointCloud::FromLegacy(cloud_legacy, open3d::core::Float32, open3d::core::Device("CUDA:0")));  // Runtime error: Unsupported device "CUDA:0". Set BUILD_CUDA_MODULE=ON to compile for CUDA support and BUILD_SYCL_MODULE=ON to compile for SYCL support.
	}

	{
		open3d::core::Tensor t1;
		open3d::core::Tensor t2(open3d::core::SizeVector({ 2, 3, 4, 5 }), open3d::core::Float32);
		open3d::core::Tensor t_cpu(open3d::core::SizeVector({ 2, 3, 4, 5 }), open3d::core::Float32, open3d::core::Device("CPU:0"));
		//open3d::core::Tensor t_gpu(open3d::core::SizeVector({ 2, 3, 4, 5 }), open3d::core::Float32, open3d::core::Device("CUDA:0"));  // Runtime error: Unsupported device "CUDA:0". Set BUILD_CUDA_MODULE=ON to compile for CUDA support and BUILD_SYCL_MODULE=ON to compile for SYCL support.
	}
}

void estimate_transformation_test()
{
	// 8 vertices on a cube.
	const std::vector<Eigen::Vector3d> cube_vertices = {
		Eigen::Vector3d(0, 8, 8),
		Eigen::Vector3d(0, 0, 8),
		Eigen::Vector3d(0, 0, 0),
		Eigen::Vector3d(0, 8, 0),
		Eigen::Vector3d(8, 8, 8),
		Eigen::Vector3d(8, 0, 8),
		Eigen::Vector3d(8, 0, 0),
		Eigen::Vector3d(8, 8, 0)
	};

	// Coordinate frames: (x, y, z, theta z (deg)).
	const std::vector<std::pair<Eigen::Vector3d, double> > coord_frames = {
		/*
		std::make_pair(Eigen::Vector3d(-8, 8, 0), -60 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-10, 4, 0), -30 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-12, 0, 0), 0 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-10, -4, 0), 30 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-8, -8, 0), 60 * M_PI / 180.0)
		*/
		std::make_pair(Eigen::Vector3d(-12, 0, 0), 0 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-10, -4, 0), 30 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-8, -8, 0), 60 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(-4, -12, 0), 75 * M_PI / 180.0),
		std::make_pair(Eigen::Vector3d(0, -16, 0), 80 * M_PI / 180.0)
	};

	std::vector<std::vector<Eigen::Vector3d> > cubes_relative;
	for (auto fit = coord_frames.begin(); fit != coord_frames.end(); ++fit)
	{
		const Eigen::Transform<double, 3, Eigen::Affine> t = Eigen::Translation3d(fit->first) * Eigen::AngleAxisd(fit->second, Eigen::Vector3d(0, 0, 1));

		std::vector<Eigen::Vector3d> cube_relative;
		for (auto vit = cube_vertices.begin(); vit != cube_vertices.end(); ++vit)
			cube_relative.push_back(t.inverse() * *vit);

		cubes_relative.push_back(cube_relative);
	}

	const double noise = 0.15;
	//const double noise = 1.5;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, noise);

	std::vector<std::vector<Eigen::Vector3d> > noisy_cubes_relative;
	for (auto cit = cubes_relative.begin(); cit != cubes_relative.end(); ++cit)
	{
		std::vector<Eigen::Vector3d> noisy_cube;
		for (auto vit = cit->begin(); vit != cit->end(); ++vit)
			noisy_cube.push_back(*vit + Eigen::Vector3d(distribution(generator), distribution(generator), distribution(generator)));

		noisy_cubes_relative.push_back(noisy_cube);
	}

	//---
	const std::vector<Eigen::Vector2i> correspondences = {
		Eigen::Vector2i(0, 0),
		Eigen::Vector2i(1, 1),
		Eigen::Vector2i(2, 2),
		Eigen::Vector2i(3, 3),
		Eigen::Vector2i(4, 4),
		Eigen::Vector2i(5, 5),
		Eigen::Vector2i(6, 6),
		Eigen::Vector2i(7, 7),
	};

	std::vector<Eigen::Matrix4d> transformations_relative;
	open3d::pipelines::registration::TransformationEstimationPointToPoint p2p;
	for (size_t idx = 1; idx < noisy_cubes_relative.size(); ++idx)
	{
		const open3d::geometry::PointCloud sources(noisy_cubes_relative[idx - 1]);
		const open3d::geometry::PointCloud targets(noisy_cubes_relative[idx]);

		const Eigen::Matrix4d transformation = p2p.ComputeTransformation(targets, sources, correspondences);
		std::cout << "Transformation (estimated):\n" << transformation << std::endl;

		Eigen::Transform<double, 3, Eigen::Affine> t1 = Eigen::Translation3d(coord_frames[idx - 1].first) * Eigen::AngleAxisd(coord_frames[idx - 1].second, Eigen::Vector3d(0, 0, 1));
		Eigen::Transform<double, 3, Eigen::Affine> t2 = Eigen::Translation3d(coord_frames[idx].first) * Eigen::AngleAxisd(coord_frames[idx].second, Eigen::Vector3d(0, 0, 1));
		std::cout << "Transformation (G/T):\n" << (t1.inverse() * t2).matrix() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_open3d {

void registration();
void slam();
void odometry();

void tsdf();

}  // namespace my_open3d

int open3d_main(int argc, char *argv[])
{
	//open3d::utility::LogInfo("Info log.");

	//local::draw_example();
	//local::io_example();
	//local::octree_example();

	//local::pose_graph_example();

	//local::tensor_test();

	//-----
	//local::estimate_transformation_test();

	//-----
	my_open3d::registration();
	//my_open3d::slam();
	//my_open3d::odometry();

	//-----
	//my_open3d::tsdf();

	return 0;
}
