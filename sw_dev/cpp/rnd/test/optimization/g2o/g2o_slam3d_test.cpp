//#include "stdafx.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <open3d/Open3D.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
//#include <g2o/types/slam3d/edge_pointxyz.h>


namespace {
namespace local {

//G2O_USE_ROBUST_KERNEL(RobustKernelHuber)
//G2O_USE_ROBUST_KERNEL(RobustKernelPseudoHuber)
//G2O_USE_ROBUST_KERNEL(RobustKernelCauchy)
//G2O_USE_ROBUST_KERNEL(RobustKernelGemanMcClure)
//G2O_USE_ROBUST_KERNEL(RobustKernelWelsch)
//G2O_USE_ROBUST_KERNEL(RobustKernelFair)
//G2O_USE_ROBUST_KERNEL(RobustKernelTukey)
//G2O_USE_ROBUST_KERNEL(RobustKernelSaturated)
//G2O_USE_ROBUST_KERNEL(RobustKernelDCS)

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://goodgodgd.github.io/ian-flow/archivers/how-to-use-g2o
void simple_slam3d_test()
{
	using block_solver_type = g2o::BlockSolver_6_3;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >;
	//using block_solver_type = g2o::BlockSolverX;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >;

	//using linear_solver_type = g2o::LinearSolverDense<block_solver_type::PoseMatrixType>;
	using linear_solver_type = g2o::LinearSolverEigen<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCholmod<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCSparse<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverPCG<block_solver_type::PoseMatrixType>;

	auto linearSolver = g2o::make_unique<linear_solver_type>();
	//linearSolver->setBlockOrdering(false);

	//auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<block_solver_type>(std::move(linearSolver)));
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<block_solver_type>(std::move(linearSolver)));

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	std::vector<g2o::SE3Quat> gt_poses;

	// Vertices.
	{
		int vertex_id = 0;

		{
			// First vertex at origin.
			const Eigen::Vector3d pos(0, 0, 0);
			Eigen::Quaterniond quat;
			quat.setIdentity();
			const g2o::SE3Quat pose(quat, pos);

			auto v_se3 = new g2o::VertexSE3;
			v_se3->setId(vertex_id++);
			v_se3->setEstimate(pose);
			v_se3->setFixed(true);

			optimizer.addVertex(v_se3);
			gt_poses.push_back(pose);
		}

		{
			// Second vertex at (1, 0, 0).
			const Eigen::Vector3d pos(1, 0, 0);
			Eigen::Quaterniond quat;
			quat.setIdentity();
			const g2o::SE3Quat pose(quat, pos);

			auto v_se3 = new g2o::VertexSE3;
			v_se3->setId(vertex_id++);
			v_se3->setEstimate(pose);
			v_se3->setFixed(true);

			optimizer.addVertex(v_se3);
			gt_poses.push_back(pose);
		}

		const int CIRCLE_NODES = 8;
		const double CIRCLE_RADIUS = 2.0;
		const double delta_angle = 2.0 * M_PI / double(CIRCLE_NODES);
		for (int i = 0; i < CIRCLE_NODES; ++i)
		{
			const double angle(delta_angle * (i + 1));
			const Eigen::Vector3d pos(CIRCLE_RADIUS * std::sin(angle), CIRCLE_RADIUS - CIRCLE_RADIUS * std::cos(angle), 0.0);
			const Eigen::Quaterniond quat(Eigen::AngleAxisd(angle, Eigen::Vector3d(0, 0, 1)));
			const g2o::SE3Quat pose(quat, pos);

			auto v_se3 = new g2o::VertexSE3;
			v_se3->setId(vertex_id++);
			v_se3->setEstimate(pose);
			v_se3->setFixed(false);

			optimizer.addVertex(v_se3);
			gt_poses.push_back(pose);
		}
	}

	// Edges.
	{
		for (size_t i = 1; i < gt_poses.size(); ++i)
		{
			// relpose: pose[i-1] w.r.t pose[i]
			const g2o::SE3Quat relpose(gt_poses[i - 1].inverse() * gt_poses[i]);

			auto edge = new g2o::EdgeSE3;
			edge->setVertex(0, optimizer.vertices().find(i - 1)->second);
			edge->setVertex(1, optimizer.vertices().find(i)->second);
			edge->setMeasurement(relpose);
			const Eigen::MatrixXd info_matrix(Eigen::MatrixXd::Identity(6, 6) * 10.0);
			edge->setInformation(info_matrix);

			optimizer.addEdge(edge);
		}

		{
			// The last pose supposed to be the same as gt_poses[1].
			const g2o::SE3Quat relpose(gt_poses.back().inverse() * gt_poses[1]);

			auto edge = new g2o::EdgeSE3;
			edge->setVertex(0, optimizer.vertices().find(gt_poses.size() - 1)->second);
			edge->setVertex(1, optimizer.vertices().find(1)->second);
			edge->setMeasurement(relpose);
			const Eigen::MatrixXd info_matrix(Eigen::MatrixXd::Identity(6, 6) * 10.0);
			edge->setInformation(info_matrix);

			optimizer.addEdge(edge);
		}
	}

	const std::string input_filename("../before_optimzation.g2o");
	optimizer.save(input_filename.c_str());

	// Optimize.
	optimizer.initializeOptimization();
	optimizer.setVerbose(true);

	std::cout << "Optimizing..." << std::endl;
	optimizer.optimize(100);
	std::cout << "Optimized." << std::endl;

	const std::string output_filename("../after_optimzation.g2o");
	optimizer.save(output_filename.c_str());
}

// REF [site] >> https://github.com/UditSinghParihar/g2o_tutorial/tree/master/landmarkSlam
void slam3d_se3_pointxyz_test()
{
	// Prepare data.

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
	//	Camera frames wrt the world frame (T^W_C).
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

	// Relative coordinates of the cube vertices wrt each robot frame.
	std::vector<std::vector<Eigen::Vector3d> > cubes_relative;
	for (auto fit = coord_frames.begin(); fit != coord_frames.end(); ++fit)
	{
		const Eigen::Transform<double, 3, Eigen::Affine> T = Eigen::Translation3d(fit->first) * Eigen::AngleAxisd(fit->second, Eigen::Vector3d(0, 0, 1));
		const auto Tinv(T.inverse());

		std::vector<Eigen::Vector3d> cube_relative;
		for (auto vit = cube_vertices.begin(); vit != cube_vertices.end(); ++vit)
			cube_relative.push_back(Tinv * *vit);

		cubes_relative.push_back(cube_relative);
	}

	//const double noise = 1.8;
	const double noise = 0.15;
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

	// Relative transformations of the i-th robot frame wrt the (i-1)-th robot frame.
	std::vector<Eigen::Matrix4d> transformations_relative;
	open3d::pipelines::registration::TransformationEstimationPointToPoint p2p;
	for (size_t idx = 1; idx < noisy_cubes_relative.size(); ++idx)
	{
		const open3d::geometry::PointCloud sources(noisy_cubes_relative[idx - 1]);
		const open3d::geometry::PointCloud targets(noisy_cubes_relative[idx]);

		transformations_relative.push_back(p2p.ComputeTransformation(targets, sources, correspondences));

#if 0
		const Eigen::Matrix4d T_est(p2p.ComputeTransformation(targets, sources, correspondences));
		std::cout << "Transformation (estimated):\n" << T_est << std::endl;

		const Eigen::Transform<double, 3, Eigen::Affine> T1(Eigen::Translation3d(coord_frames[idx - 1].first) * Eigen::AngleAxisd(coord_frames[idx - 1].second, Eigen::Vector3d(0, 0, 1)));
		const Eigen::Transform<double, 3, Eigen::Affine> T2(Eigen::Translation3d(coord_frames[idx].first) * Eigen::AngleAxisd(coord_frames[idx].second, Eigen::Vector3d(0, 0, 1)));
		std::cout << "Transformation (G/T):\n" << (T1.inverse() * T2).matrix() << std::endl;
#endif
	}

	//--------------------
	// Optimization problem.

	using block_solver_type = g2o::BlockSolver_6_3;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >;
	//using block_solver_type = g2o::BlockSolverX;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >;

	//using linear_solver_type = g2o::LinearSolverDense<block_solver_type::PoseMatrixType>;
	using linear_solver_type = g2o::LinearSolverEigen<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCholmod<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCSparse<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverPCG<block_solver_type::PoseMatrixType>;

	auto linearSolver = g2o::make_unique<linear_solver_type>();
	//linearSolver->setBlockOrdering(false);

	//auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<block_solver_type>(std::move(linearSolver)));
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<block_solver_type>(std::move(linearSolver)));

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	//-----
	// Initial frame: (x, y, z, theta z (deg)).
	//const std::pair<Eigen::Vector3d, double> init_frame = std::make_pair(Eigen::Vector3d(-12, 0, 0), 0.0);
	const std::pair<Eigen::Vector3d, double> &init_frame = coord_frames[0];
	const Eigen::Transform<double, 3, Eigen::Affine> &Tw1 = Eigen::Translation3d(init_frame.first) * Eigen::AngleAxisd(init_frame.second, Eigen::Vector3d(0, 0, 1));

	const Eigen::Matrix4d T01(Tw1.matrix());
	const Eigen::Matrix4d T02(T01 * transformations_relative[0]);
	const Eigen::Matrix4d T03(T02 * transformations_relative[1]);
	const Eigen::Matrix4d T04(T03 * transformations_relative[2]);
	const Eigen::Matrix4d T05(T04 * transformations_relative[3]);
	const std::vector<Eigen::Matrix4d> Ts = {T01, T02, T03, T04, T05};

	int vertex_id = 1;

	// Sensor offset parameter.
	const int sensor_id = 0;
	g2o::Isometry3 sensorOffsetTransf;
	auto sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(sensorOffsetTransf);
	sensorOffset->setId(sensor_id);
	optimizer.addParameter(sensorOffset);

	// Robot pose vertices.
	bool is_first = true;
	for (auto it = Ts.begin(); it != Ts.end(); ++it)
	{
		auto robot = new g2o::VertexSE3;
		robot->setId(vertex_id++);
		robot->setEstimate(g2o::Isometry3(*it));  // Robot pose wrt the world frame.
		//robot->setMarginalized(true);
		if (is_first)
		{
			robot->setFixed(true);
			is_first = false;
		}

		optimizer.addVertex(robot);
	}

	// Landmark(cube) vertices.
	const std::vector<Eigen::Vector3d> &noisy_cube = noisy_cubes_relative.front();
	const auto &TT = Ts.front();
	for (auto it = noisy_cube.begin(); it != noisy_cube.end(); ++it)
	{
		auto landmark = new g2o::VertexPointXYZ;
		landmark->setId(vertex_id++);
		//const auto t(TT * Eigen::Vector4d(it->x(), it->y(), it->z(), 1.0));  // Error.
		const Eigen::Vector4d t(TT * Eigen::Vector4d(it->x(), it->y(), it->z(), 1.0));
		landmark->setEstimate(t.head<3>());  // Vertex coordinates wrt the world frame.
		//landmark->setMarginalized(true);

		optimizer.addVertex(landmark);
	}

	// Odometry constraint edges.
	for (size_t idx = 0; idx < transformations_relative.size(); ++idx)
	{
		const Eigen::Matrix<double, 6, 6> information_matrix(Eigen::DiagonalMatrix<double, 6>(20.0, 20.0, 20.0, 20.0, 20.0, 20.0));

		auto odometry = new g2o::EdgeSE3;
		//odometry->setVertex(0, optimizer.vertex(int(idx + 1)));  // Robot ID.
		//odometry->setVertex(0, optimizer.vertex(int(idx + 2)));  // Robot ID.
		odometry->vertices()[0] = optimizer.vertex(int(idx + 1));  // Robot ID.
		odometry->vertices()[1] = optimizer.vertex(int(idx + 2));  // Robot ID.
		odometry->setMeasurement(g2o::Isometry3(transformations_relative[idx]));  // Relative pose of the i-th robot frame wrt the (i-1)-th robot frame.
		odometry->setInformation(information_matrix);

#if 0
		//auto robust_kernel = g2o::RobustKernelFactory::instance()->construct("Huber");
		auto robust_kernel = new g2o::RobustKernelHuber;
		robust_kernel->setDelta(std::sqrt(5.991));  // 95% CI.
		odometry->setRobustKernel(robust_kernel);
#endif

		optimizer.addEdge(odometry);
	}

	// Landmark(cube) observation edges.
	for (size_t idx = 0; idx < noisy_cubes_relative.size(); ++idx)
	{
		const Eigen::Matrix3d information_matrix(Eigen::DiagonalMatrix<double, 3>(40.0, 40.0, 40.0));

		for (size_t vidx = 0; vidx < noisy_cubes_relative[idx].size(); ++vidx)
		{
			auto observation = new g2o::EdgeSE3PointXYZ;
			//observation->setVertex(0, optimizer.vertex(int(idx + 1)));  // Robot ID.
			//observation->setVertex(1, optimizer.vertex(int(vidx + 6)));  // Landmark ID.
			observation->vertices()[0] = optimizer.vertex(int(idx + 1));  // Robot ID.
			observation->vertices()[1] = optimizer.vertex(int(vidx + 6));  // Landmark ID.
			observation->setMeasurement(noisy_cubes_relative[idx][vidx]);  // Relative coordinates of each vertex wrt the robot frame.
			observation->setInformation(information_matrix);
			observation->setParameterId(0, sensor_id);  // Required.

#if 0
			//auto robust_kernel = g2o::RobustKernelFactory::instance()->construct("Huber");
			auto robust_kernel = new g2o::RobustKernelHuber;
			robust_kernel->setDelta(std::sqrt(5.991));  // 95% CI.
			observation->setRobustKernel(robust_kernel);
#endif

			optimizer.addEdge(observation);
		}
	}

	std::cout << "#vertices = " << optimizer.vertices().size() << std::endl;
	std::cout << "#edges = " << optimizer.edges().size() << std::endl;
	std::cout << "Chi2 = " << optimizer.chi2() << std::endl;
	std::cout << "Max dimension = " << optimizer.maxDimension() << std::endl;
#if 0
	std::cout << "Dimensions = ";
	std::copy(optimizer.dimensions().begin(), optimizer.dimensions().end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;
#endif

	const std::string output_before_filename("../slam3d_se3_pointxyz_before_pgo.g2o");
	if (!optimizer.save(output_before_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_before_filename << std::endl;
		return;
	}

	//-----
	// Optimize.
	const int max_iterations = 100;
	const bool online = false;
	const bool verbose = true;

	if (!optimizer.initializeOptimization())
	{
		std::cerr << "Optimizer not initialized." << std::endl;
		return;
	}
	optimizer.setVerbose(verbose);

	std::cout << "Optimizing..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	const auto num_iterations = optimizer.optimize(max_iterations, online);
	std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	std::cout << "#iterations = " << num_iterations << std::endl;

	//-----
	const std::string output_filename("../slam3d_se3_pointxyz_after_pgo.g2o");
	if (!optimizer.save(output_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_filename << std::endl;
		return;
	}

	// Show results.
	for (int vid = 1; vid <= 5; ++vid)
	{
		const auto &T_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(vid))->estimate();
		std::cout << "Vertex #" << vid << std::endl;
		std::cout << T_est.matrix() << std::endl;
	}
	for (int vid = 6; vid <= 13; ++vid)
	{
		const auto &l_est = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(vid))->estimate();
		std::cout << "Vertex #" << vid << ": " << l_est.transpose() << std::endl;
	}

	// Free the graph memory.
	optimizer.clear();
}

// REF [site] >> https://github.com/UditSinghParihar/g2o_tutorial/tree/master/landmarkSlam
void slam3d_se3_test()
{
	// Prepare data.

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
	//	Camera frames wrt the world frame (T^W_C).
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

	// Relative coordinates of the cube vertices wrt each robot frame.
	std::vector<std::vector<Eigen::Vector3d> > cubes_relative;
	for (auto fit = coord_frames.begin(); fit != coord_frames.end(); ++fit)
	{
		const Eigen::Transform<double, 3, Eigen::Affine> T = Eigen::Translation3d(fit->first) * Eigen::AngleAxisd(fit->second, Eigen::Vector3d(0, 0, 1));
		const auto Tinv(T.inverse());

		std::vector<Eigen::Vector3d> cube_relative;
		for (auto vit = cube_vertices.begin(); vit != cube_vertices.end(); ++vit)
			cube_relative.push_back(Tinv * *vit);

		cubes_relative.push_back(cube_relative);
	}

	//const double noise = 1.8;
	const double noise = 0.15;
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

	// Relative transformations of the i-th robot frame wrt the (i-1)-th robot frame.
	std::vector<Eigen::Matrix4d> transformations_relative;
	open3d::pipelines::registration::TransformationEstimationPointToPoint p2p;
	for (size_t idx = 1; idx < noisy_cubes_relative.size(); ++idx)
	{
		const open3d::geometry::PointCloud sources(noisy_cubes_relative[idx - 1]);
		const open3d::geometry::PointCloud targets(noisy_cubes_relative[idx]);

		transformations_relative.push_back(p2p.ComputeTransformation(targets, sources, correspondences));

#if 0
		const Eigen::Matrix4d T_est(p2p.ComputeTransformation(targets, sources, correspondences));
		std::cout << "Transformation (estimated):\n" << T_est << std::endl;

		const Eigen::Transform<double, 3, Eigen::Affine> T1(Eigen::Translation3d(coord_frames[idx - 1].first) * Eigen::AngleAxisd(coord_frames[idx - 1].second, Eigen::Vector3d(0, 0, 1)));
		const Eigen::Transform<double, 3, Eigen::Affine> T2(Eigen::Translation3d(coord_frames[idx].first) * Eigen::AngleAxisd(coord_frames[idx].second, Eigen::Vector3d(0, 0, 1)));
		std::cout << "Transformation (G/T):\n" << (T1.inverse() * T2).matrix() << std::endl;
#endif
	}

	//--------------------
	// Optimization problem.

	using block_solver_type = g2o::BlockSolver_6_3;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >;
	//using block_solver_type = g2o::BlockSolverX;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >;

	//using linear_solver_type = g2o::LinearSolverDense<block_solver_type::PoseMatrixType>;
	using linear_solver_type = g2o::LinearSolverEigen<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCholmod<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCSparse<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverPCG<block_solver_type::PoseMatrixType>;

	auto linearSolver = g2o::make_unique<linear_solver_type>();
	//linearSolver->setBlockOrdering(false);

	//auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<block_solver_type>(std::move(linearSolver)));
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<block_solver_type>(std::move(linearSolver)));

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	//-----
	// Initial frame: (x, y, z, theta z (deg)).
	//const std::pair<Eigen::Vector3d, double> init_frame = std::make_pair(Eigen::Vector3d(-12, 0, 0), 0.0);
	const std::pair<Eigen::Vector3d, double> &init_frame = coord_frames[0];
	const Eigen::Transform<double, 3, Eigen::Affine> &Tw1 = Eigen::Translation3d(init_frame.first) * Eigen::AngleAxisd(init_frame.second, Eigen::Vector3d(0, 0, 1));

	const Eigen::Matrix4d T01(Tw1.matrix());
	const Eigen::Matrix4d T02(T01 * transformations_relative[0]);
	const Eigen::Matrix4d T03(T02 * transformations_relative[1]);
	const Eigen::Matrix4d T04(T03 * transformations_relative[2]);
	const Eigen::Matrix4d T05(T04 * transformations_relative[3]);
	const std::vector<Eigen::Matrix4d> Ts = {T01, T02, T03, T04, T05};

	int vertex_id = 1;

	// Robot pose vertices.
	bool is_first = true;
	for (auto it = Ts.begin(); it != Ts.end(); ++it)
	{
		auto robot = new g2o::VertexSE3;
		robot->setId(vertex_id++);
		robot->setEstimate(g2o::Isometry3(*it));  // Robot pose wrt the world frame.
		//robot->setMarginalized(true);
		if (is_first)
		{
			robot->setFixed(true);
			is_first = false;
		}

		optimizer.addVertex(robot);
	}

	// Landmark(cube) vertices.
	const std::vector<Eigen::Vector3d> &noisy_cube = noisy_cubes_relative.front();
	const auto &TT = Ts.front();
	for (auto it = noisy_cube.begin(); it != noisy_cube.end(); ++it)
	{
		auto landmark = new g2o::VertexSE3;
		landmark->setId(vertex_id++);
		//const auto t(TT * Eigen::Vector4d(it->x(), it->y(), it->z(), 1.0));  // Error.
		const Eigen::Vector4d t(TT * Eigen::Vector4d(it->x(), it->y(), it->z(), 1.0));
		landmark->setEstimate(g2o::Isometry3(Eigen::Translation3d(t.head<3>()) * Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0)));  // Vertex pose wrt the world frame.
		//landmark->setMarginalized(true);

		optimizer.addVertex(landmark);
	}

	// Odometry constraint edges.
	for (size_t idx = 0; idx < transformations_relative.size(); ++idx)
	{
		const Eigen::Matrix<double, 6, 6> information_matrix(Eigen::DiagonalMatrix<double, 6>(20.0, 20.0, 20.0, 20.0, 20.0, 20.0));

		auto odometry = new g2o::EdgeSE3;
		//odometry->setVertex(0, optimizer.vertex(int(idx + 1)));  // Robot ID.
		//odometry->setVertex(0, optimizer.vertex(int(idx + 2)));  // Robot ID.
		odometry->vertices()[0] = optimizer.vertex(int(idx + 1));  // Robot ID.
		odometry->vertices()[1] = optimizer.vertex(int(idx + 2));  // Robot ID.
		odometry->setMeasurement(g2o::Isometry3(transformations_relative[idx]));  // Relative pose of the i-th robot frame wrt the (i-1)-th robot frame.
		odometry->setInformation(information_matrix);

#if 0
		//auto robust_kernel = g2o::RobustKernelFactory::instance()->construct("Huber");
		auto robust_kernel = new g2o::RobustKernelHuber;
		robust_kernel->setDelta(std::sqrt(5.991));  // 95% CI.
		odometry->setRobustKernel(robust_kernel);
#endif

		optimizer.addEdge(odometry);
	}

	// Landmark(cube) observation edges.
	for (size_t idx = 0; idx < noisy_cubes_relative.size(); ++idx)
	{
		const Eigen::Matrix<double, 6, 6> information_matrix(Eigen::DiagonalMatrix<double, 6>(40.0, 40.0, 40.0, 0.000001, 0.000001, 0.000001));

		for (size_t vidx = 0; vidx < noisy_cubes_relative[idx].size(); ++vidx)
		{
			auto observation = new g2o::EdgeSE3;
			//observation->setVertex(0, optimizer.vertex(int(idx + 1)));  // Robot ID.
			//observation->setVertex(1, optimizer.vertex(int(vidx + 6)));  // Landmark ID.
			observation->vertices()[0] = optimizer.vertex(int(idx + 1));  // Robot ID.
			observation->vertices()[1] = optimizer.vertex(int(vidx + 6));  // Landmark ID.
			observation->setMeasurement(g2o::Isometry3(Eigen::Translation3d(noisy_cubes_relative[idx][vidx]) * Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0)));  // Relative pose of each vertex wrt the robot frame.
			observation->setInformation(information_matrix);
			//observation->setParameterId(0, sensor_id);  // Not required.

#if 0
			//auto robust_kernel = g2o::RobustKernelFactory::instance()->construct("Huber");
			auto robust_kernel = new g2o::RobustKernelHuber;
			robust_kernel->setDelta(std::sqrt(5.991));  // 95% CI.
			observation->setRobustKernel(robust_kernel);
#endif

			optimizer.addEdge(observation);
		}
	}

	std::cout << "#vertices = " << optimizer.vertices().size() << std::endl;
	std::cout << "#edges = " << optimizer.edges().size() << std::endl;
	std::cout << "Chi2 = " << optimizer.chi2() << std::endl;
	std::cout << "Max dimension = " << optimizer.maxDimension() << std::endl;
#if 0
	std::cout << "Dimensions = ";
	std::copy(optimizer.dimensions().begin(), optimizer.dimensions().end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;
#endif

	const std::string output_before_filename("../slam3d_se3_before_pgo.g2o");
	if (!optimizer.save(output_before_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_before_filename << std::endl;
		return;
	}

	//-----
	// Optimize.
	const int max_iterations = 100;
	const bool online = false;
	const bool verbose = true;

	if (!optimizer.initializeOptimization())
	{
		std::cerr << "Optimizer not initialized." << std::endl;
		return;
	}
	optimizer.setVerbose(verbose);

	std::cout << "Optimizing..." << std::endl;
	const auto start_time(std::chrono::high_resolution_clock::now());
	const auto num_iterations = optimizer.optimize(max_iterations, online);
	std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " msecs." << std::endl;
	std::cout << "#iterations = " << num_iterations << std::endl;

	//-----
	const std::string output_filename("../slam3d_se3_after_pgo.g2o");
	if (!optimizer.save(output_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_filename << std::endl;
		return;
	}

	// Show results.
	for (int vid = 1; vid <= 13; ++vid)
	{
		const auto &T_est = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(vid))->estimate();
		std::cout << "Vertex #" << vid << std::endl;
		std::cout << T_est.matrix() << std::endl;
	}

	// Free the graph memory.
	optimizer.clear();
}

}  // namespace my_g2o
