#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy, time
import numpy as np
import open3d as o3d

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
def icp_registration_tutorial():
	def draw_registration_result(source, target, transformation):
		source_temp = copy.deepcopy(source)
		target_temp = copy.deepcopy(target)
		source_temp.paint_uniform_color([1, 0.706, 0])
		target_temp.paint_uniform_color([0, 0.651, 0.929])
		source_temp.transform(transformation)
		o3d.visualization.draw_geometries(
			[source_temp, target_temp],
			zoom=0.4459,
			front=[0.9288, -0.2951, -0.2242],
			lookat=[1.6784, 2.0612, 1.4451],
			up=[-0.3402, -0.9189, -0.1996]
		)

	source = o3d.io.read_point_cloud("./test_data/ICP/cloud_bin_0.pcd")
	target = o3d.io.read_point_cloud("./test_data/ICP/cloud_bin_1.pcd")
	threshold = 0.02

	print("#source points = {}.".format(len(source.points)))
	print("#target points = {}.".format(len(target.points)))

	trans_init = np.asarray([
		[0.862, 0.011, -0.507, 0.5],
		[-0.139, 0.967, -0.215, 0.7],
		[0.487, 0.255, 0.835, -1.4],
		[0.0, 0.0, 0.0, 1.0]
	])
	draw_registration_result(source, target, trans_init)

	print("Initial alignment")
	evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
	print(evaluation)

	# Point-to-point ICP.
	print("Apply point-to-point ICP")
	start_time = time.time()
	reg_p2p = o3d.pipelines.registration.registration_icp(
		source, target, threshold, trans_init,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
	)
	print("Point-to-point ICP applied: {} secs.".format(time.time() - start_time))
	print(reg_p2p)
	print("Transformation is:")
	print(reg_p2p.transformation)
	draw_registration_result(source, target, reg_p2p.transformation)

	# Point-to-plane ICP.
	print("Apply point-to-plane ICP")
	start_time = time.time()
	reg_p2l = o3d.pipelines.registration.registration_icp(
		source, target, threshold, trans_init,
		o3d.pipelines.registration.TransformationEstimationPointToPlane(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
	)
	print("Point-to-plane ICP applied: {} secs.".format(time.time() - start_time))
	print(reg_p2l)
	print("Transformation is:")
	print(reg_p2l.transformation)
	draw_registration_result(source, target, reg_p2l.transformation)

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
def colored_point_cloud_registration_tutorial():
	def draw_registration_result_original_color(source, target, transformation):
		source_temp = copy.deepcopy(source)
		source_temp.transform(transformation)
		o3d.visualization.draw_geometries(
			[source_temp, target],
			zoom=0.5,
			front=[-0.2458, -0.8088, 0.5342],
			lookat=[1.7745, 2.2305, 0.9787],
			up=[0.3109, -0.5878, -0.7468]
		)

	print("1. Load two point clouds and show initial pose")
	source = o3d.io.read_point_cloud("./test_data/ColoredICP/frag_115.ply")
	target = o3d.io.read_point_cloud("./test_data/ColoredICP/frag_116.ply")

	print("#source points = {}.".format(len(source.points)))
	print("#target points = {}.".format(len(target.points)))

	# Draw initial alignment.
	current_transformation = np.identity(4)
	draw_registration_result_original_color(source, target, current_transformation)

	# Point-to-plane ICP.
	current_transformation = np.identity(4)
	print("2. Point-to-plane ICP registration is applied on original point")
	print("   clouds to refine the alignment. Distance threshold 0.02.")
	start_time = time.time()
	result_icp = o3d.pipelines.registration.registration_icp(
		source, target, 0.02, current_transformation,
		o3d.pipelines.registration.TransformationEstimationPointToPlane()
	)
	print("Point-to-plane ICP applied: {} secs.".format(time.time() - start_time))
	print(result_icp)
	draw_registration_result_original_color(source, target, result_icp.transformation)

	# Colored point cloud registration.
	#	"Colored Point Cloud Registration Revisited", ICCV 2017.
	voxel_radius = [0.04, 0.02, 0.01]
	max_iter = [50, 30, 14]
	current_transformation = np.identity(4)
	print("3. Colored point cloud registration")
	start_time = time.time()
	for scale in range(3):
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		print([iter, radius, scale])

		print("3-1. Downsample with a voxel size %.2f" % radius)
		source_down = source.voxel_down_sample(radius)
		target_down = target.voxel_down_sample(radius)

		print("3-2. Estimate normal.")
		source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
		target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

		print("3-3. Applying colored point cloud registration")
		result_icp = o3d.pipelines.registration.registration_colored_icp(
			source_down, target_down, radius, current_transformation,
			o3d.pipelines.registration.TransformationEstimationForColoredICP(),
			o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter)
		)
		current_transformation = result_icp.transformation
		print(result_icp)
	print("Colored point cloud registration applied: {} secs.".format(time.time() - start_time))
	draw_registration_result_original_color(source, target, result_icp.transformation)

# REF [site] >> http://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html
def t_icp_registration_tutorial():
	def draw_registration_result(source, target, transformation):
		source_temp = source.clone()
		target_temp = target.clone()

		source_temp.transform(transformation)

		# This is patched version for tutorial rendering.
		# Use 'draw' function for you application.
		o3d.visualization.draw_geometries(
			[source_temp.to_legacy(), target_temp.to_legacy()],
			zoom=0.4459,
			front=[0.9288, -0.2951, -0.2242],
			lookat=[1.6784, 2.0612, 1.4451],
			up=[-0.3402, -0.9189, -0.1996]
		)

	demo_icp_pcds = o3d.data.DemoICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

	# For Colored-ICP 'colors' attribute must be of the same dtype as 'positions' and 'normals' attribute.
	source.point["colors"] = source.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
	target.point["colors"] = target.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

	# Initial guess transform between the two point-cloud.
	# ICP algortihm requires a good initial alignment to converge efficiently.
	trans_init = np.asarray([
		[0.862, 0.011, -0.507, 0.5],
		[-0.139, 0.967, -0.215, 0.7],
		[0.487, 0.255, 0.835, -1.4],
		[0.0, 0.0, 0.0, 1.0]
	])

	draw_registration_result(source, target, trans_init)

	#--------------------
	# Vanilla ICP example.

	# Input point-clouds.
	demo_icp_pcds = o3d.data.DemoICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

	# Select the 'Estimation Method', and 'Robust Kernel' (for outlier-rejection).
	# Estimation method:
	#	Point-to-Point ICP: o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
	#	Point-to-Plane ICP: o3d.t.pipelines.registration.TransformationEstimationPointToPlane(robust_kernel)
	#	Colored ICP: o3d.t.pipelines.registration.TransformationEstimationForColoredICP(robust_kernel, lambda)
	#	Generalized ICP: o3d.t.pipelines.registration.TransformationEstimationForGeneralizedICP(robust_kernel, epsilon)
	# Robust kernel:
	#	robust_kernel = o3d.t.pipelines.registration.robust_kernel.RobustKernel(method, scale, shape)
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.L2Loss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.L1Loss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.HuberLoss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.CauchyLoss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.GMLoss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss
	#		o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.GeneralizedLoss

	# Search distance for nearest neighbour search [hybrid-search is used].
	max_correspondence_distance = 0.07
	# Select the 'estimation method', and 'robust kernel' (for outlier-rejection).
	estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
	# Convergence-criteria for vanilla ICP.
	criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=50)
	# Down-sampling voxel-size.
	voxel_size = 0.025

	# Save iteration wise 'fitness', 'inlier_rmse', etc. to analyse and tune result.
	save_loss_log = True

	# Get iteration-wise registration result using callback lambda function.
	callback_after_iteration = lambda updated_result_dict : print(
		"Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
			updated_result_dict["iteration_index"].item(),
			updated_result_dict["fitness"].item(),
			updated_result_dict["inlier_rmse"].item()
		)
	)

	# Initial alignment or source to target transform.
	init_source_to_target = np.asarray([
		[0.862, 0.011, -0.507, 0.5],
		[-0.139, 0.967, -0.215, 0.7],
		[0.487, 0.255, 0.835, -1.4],
		[0.0, 0.0, 0.0, 1.0]
	])

	s = time.time()
	registration_icp = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size, callback_after_iteration
	)
	icp_time = time.time() - s
	print("Time taken by ICP: ", icp_time)
	print("Inlier Fitness: ", registration_icp.fitness)
	print("Inlier RMSE: ", registration_icp.inlier_rmse)

	draw_registration_result(source, target, registration_icp.transformation)

	#-----
	# Now let's try with poor initial initialisation.
	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
	max_correspondence_distance = 0.07

	s = time.time()
	registration_icp = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size
	)
	icp_time = time.time() - s
	print("Time taken by ICP: ", icp_time)
	print("Inlier Fitness: ", registration_icp.fitness)
	print("Inlier RMSE: ", registration_icp.inlier_rmse)

	draw_registration_result(source, target, registration_icp.transformation)

	#-----
	# As we can see, poor initial alignment might fail ICP convergence.
	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.float32)
	max_correspondence_distance = 0.5

	s = time.time()
	# It is highly recommended to down-sample the point-cloud before using ICP algorithm, for better performance.
	registration_icp = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size
	)
	icp_time = time.time() - s
	print("Time taken by ICP: ", icp_time)
	print("Inlier Fitness: ", registration_icp.fitness)
	print("Inlier RMSE: ", registration_icp.inlier_rmse)

	draw_registration_result(source, target, registration_icp.transformation)

	#--------------------
	# Multi-scale ICP example.

	# Input point-clouds.
	demo_icp_pcds = o3d.data.DemoICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

	# For multi-scale ICP (o3d.utility.DoubleVector):
	# 'max_correspondence_distances' is proportianal to the resolution or the 'voxel_sizes'.
	# In general it is recommended to use values between 1x - 3x of the corresponding 'voxel_sizes'.
	# We may have a higher value of the 'max_correspondence_distances' for the first coarse scale, as it is not much expensive, and gives us more tolerance to initial alignment.
	max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
	# List of convergence-criteria for multi-scale ICP:
	# We can control 'ConvergenceCriteria' of each 'scale' individually.
	# We want to keep 'relative_fitness' and 'relative_rmse' high (more error tolerance) for initial scales, i.e. we will be happy to consider ICP converged, when difference between 2 successive iterations for that scale is smaller than this value.
	# We expect less accuracy (more error tolerance) initial coarse-scale iteration, and want our later scale convergence to be more accurate (less error tolerance).
	criteria_list = [
		o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=20),
		o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
		o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
	]
	# Lower 'voxel_size' is equivalent to higher resolution, and we want to perform iterations from coarse to dense resolution, therefore 'voxel_sizes' must be in strictly decressing order.
	voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
	# Select the 'estimation method', and 'robust kernel' (for outlier-rejection).
	estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

	# Save iteration wise 'fitness', 'inlier_rmse', etc. to analyse and tune result.
	callback_after_iteration = lambda loss_log_map : print(
		"Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
			loss_log_map["iteration_index"].item(),
			loss_log_map["scale_index"].item(),
			loss_log_map["scale_iteration_index"].item(),
			loss_log_map["fitness"].item(),
			loss_log_map["inlier_rmse"].item()
		)
	)

	# Initial alignment or source to target transform.
	init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

	# Setting Verbosity to Debug, helps in fine-tuning the performance.
	#o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

	s = time.time()
	registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(
		source, target,
		voxel_sizes, criteria_list,
		max_correspondence_distances,
		init_source_to_target, estimation,
		callback_after_iteration
	)
	ms_icp_time = time.time() - s
	print("Time taken by Multi-Scale ICP: ", ms_icp_time)
	print("Inlier Fitness: ", registration_ms_icp.fitness)
	print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)

	draw_registration_result(source, target, registration_ms_icp.transformation)

	#-----
	# Multi-scale ICP on CUDA device example.

	# The algorithm runs on the same device as the source and target point-cloud.
	source_cuda = source.cuda(0)
	target_cuda = target.cuda(0)

	s = time.time()
	registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(
		source_cuda, target_cuda,
		voxel_sizes, criteria_list,
		max_correspondence_distances,
		init_source_to_target, estimation
	)
	ms_icp_time = time.time() - s
	print("Time taken by Multi-Scale ICP: ", ms_icp_time)
	print("Inlier Fitness: ", registration_ms_icp.fitness)
	print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)

	draw_registration_result(source.cpu(), target.cpu(), registration_ms_icp.transformation)

	#--------------------
	# Case of no correspondences.
	#	In case of no_correspondences the fitness and inlier_rmse is 0.
	max_correspondence_distance = 0.02
	init_source_to_target = np.asarray([
		[1.0, 0.0, 0.0, 5],
		[0.0, 1.0, 0.0, 7],
		[0.0, 0.0, 1.0, 10],
		[0.0, 0.0, 0.0, 1.0]
	])

	registration_icp = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target
	)

	print("Inlier Fitness: ", registration_icp.fitness)
	print("Inlier RMSE: ", registration_icp.inlier_rmse)
	print("Transformation: \n", registration_icp.transformation)

	if registration_icp.fitness == 0 and registration_icp.inlier_rmse == 0:
		print("ICP Convergence Failed, as no correspondence were found")

	#-----
	# Information matrix.
	#	Information matrix gives us futher information about how well the point-clouds are aligned.
	information_matrix = o3d.t.pipelines.registration.get_information_matrix(
		source, target,
		max_correspondence_distances[2],
		registration_ms_icp.transformation
	)

	print("Information Matrix:\n", information_matrix)

	#-----
	# Initial alignment.
	demo_icp_pcds = o3d.data.DemoICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

	# Initial guess transform between the two point-cloud.
	# ICP algortihm requires a good initial alignment to converge efficiently.
	trans_init = np.asarray([
		[0.862, 0.011, -0.507, 0.5],
		[-0.139, 0.967, -0.215, 0.7],
		[0.487, 0.255, 0.835, -1.4],
		[0.0, 0.0, 0.0, 1.0]
	])

	draw_registration_result(source, target, trans_init)

	# Search distance for nearest neighbour search [hybrid-search is used].
	max_correspondence_distance = 0.02

	print("Initial Alignment")
	evaluation = o3d.t.pipelines.registration.evaluate_registration(source, target, max_correspondence_distance, trans_init)

	print("Fitness: ", evaluation.fitness)
	print("Inlier RMSE: ", evaluation.inlier_rmse)

	#--------------------
	# Point-to-point ICP registration.

	# Input point-clouds.
	demo_icp_pcds = o3d.data.DemoICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])

	# Select the 'estimation method', and 'robust kernel' (for outlier-rejection).
	estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
	# Search distance for nearest neighbour search [hybrid-search is used].
	max_correspondence_distance = 0.02
	# Convergence-Criteria for Vanilla ICP.
	criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=30)
	# Down-sampling voxel-size.
	#	If voxel_size < 0, original scale is used.
	voxel_size = -1
	# Initial alignment or source to target transform.
	init_source_to_target = trans_init

	# Save iteration wise 'fitness', 'inlier_rmse', etc. to analyse and tune result.
	save_loss_log = True

	print("Apply Point-to-Point ICP")
	s = time.time()
	reg_point_to_point = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size
	)
	icp_time = time.time() - s
	print("Time taken by Point-To-Point ICP: ", icp_time)
	print("Fitness: ", reg_point_to_point.fitness)
	print("Inlier RMSE: ", reg_point_to_point.inlier_rmse)

	draw_registration_result(source, target, reg_point_to_point.transformation)

	#-----
	# The fitness score increases from 0.174722 to 0.372474.
	# The inlier_rmse reduces from 0.011771 to 0.007761.
	# By default, icp runs until convergence or reaches a maximum number of iterations (30 by default).
	# It can be changed to allow more computation time and to improve the results further.
	criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=1000)

	print("Apply Point-to-Point ICP")
	s = time.time()
	reg_point_to_point = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size
	)
	icp_time = time.time() - s
	print("Time taken by Point-To-Point ICP: ", icp_time)
	print("Fitness: ", reg_point_to_point.fitness)
	print("Inlier RMSE: ", reg_point_to_point.inlier_rmse)

	draw_registration_result(source, target, reg_point_to_point.transformation)

	#--------------------
	# Point-to-plane ICP registration.

	estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
	criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0000001, relative_rmse=0.0000001, max_iteration=30)

	print("Apply Point-to-Plane ICP")
	s = time.time()
	reg_point_to_plane = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation, criteria,
		voxel_size
	)
	icp_time = time.time() - s
	print("Time taken by Point-To-Plane ICP: ", icp_time)
	print("Fitness: ", reg_point_to_plane.fitness)
	print("Inlier RMSE: ", reg_point_to_plane.inlier_rmse)

	draw_registration_result(source, target, reg_point_to_plane.transformation)

	#--------------------
	# Colored ICP registration.

	# Overriding visualization function, according to best camera view for colored-icp sample data.
	def draw_registration_result(source, target, transformation):
		source_temp = source.clone()
		target_temp = target.clone()

		source_temp.transform(transformation)

		# This is patched version for tutorial rendering.
		# Use 'draw' function for you application.
		o3d.visualization.draw_geometries(
			[source_temp.to_legacy(), target_temp.to_legacy()],
			zoom=0.5,
			front=[-0.2458, -0.8088, 0.5342],
			lookat=[1.7745, 2.2305, 0.9787],
			up=[0.3109, -0.5878, -0.7468]
		)

	print("Load two point clouds and show initial pose")
	demo_cicp_pcds = o3d.data.DemoColoredICPPointClouds()
	source = o3d.t.io.read_point_cloud(demo_cicp_pcds.paths[0])
	target = o3d.t.io.read_point_cloud(demo_cicp_pcds.paths[1])

	# For Colored-ICP 'colors' attribute must be of the same dtype as 'positions' and 'normals' attribute.
	source.point["colors"] = source.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
	target.point["colors"] = target.point["colors"].to(o3d.core.Dtype.Float32) / 255.0

	# Draw initial alignment.
	current_transformation = np.identity(4)
	draw_registration_result(source, target, current_transformation)

	# Setting baseline with point-to-plane registration.
	#	We first run Point-to-plane ICP as a baseline approach.
	#	The visualization below shows misaligned green triangle textures.
	#	This is because a geometric constraint does not prevent two planar surfaces from slipping.
	estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
	max_correspondence_distance = 0.02
	init_source_to_target = np.identity(4)

	print("Apply Point-to-Plane ICP")
	s = time.time()
	reg_point_to_plane = o3d.t.pipelines.registration.icp(
		source, target,
		max_correspondence_distance,
		init_source_to_target, estimation
	)
	icp_time = time.time() - s
	print("Time taken by Point-To-Plane ICP: ", icp_time)
	print("Fitness: ", reg_point_to_plane.fitness)
	print("Inlier RMSE: ", reg_point_to_plane.inlier_rmse)

	draw_registration_result(source, target, reg_point_to_plane.transformation)

	# Colored registration.
	#	"Colored Point Cloud Registration Revisited", ICCV 2017.
	estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
	init_source_to_target = np.identity(4)
	criteria_list = [
		o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001, relative_rmse=0.0001, max_iteration=50),
		o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
		o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 14)
	]
	max_correspondence_distances = o3d.utility.DoubleVector([0.08, 0.04, 0.02])
	voxel_sizes = o3d.utility.DoubleVector([0.04, 0.02, 0.01])

	print("Colored point cloud registration")
	s = time.time()
	reg_multiscale_icp = o3d.t.pipelines.registration.multi_scale_icp(
		source, target,
		voxel_sizes,
		criteria_list,
		max_correspondence_distances,
		init_source_to_target, estimation
	)
	icp_time = time.time() - s
	print("Time taken by Colored ICP: ", icp_time)
	print("Fitness: ", reg_point_to_plane.fitness)
	print("Inlier RMSE: ", reg_point_to_plane.inlier_rmse)

	draw_registration_result(source, target, reg_multiscale_icp.transformation)

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def global_registration_tutorial():
	def draw_registration_result(source, target, transformation):
		source_temp = copy.deepcopy(source)
		target_temp = copy.deepcopy(target)
		source_temp.paint_uniform_color([1, 0.706, 0])
		target_temp.paint_uniform_color([0, 0.651, 0.929])
		source_temp.transform(transformation)
		o3d.visualization.draw_geometries(
			[source_temp, target_temp],
			zoom=0.4559,
			front=[0.6452, -0.3036, -0.7011],
			lookat=[1.9892, 2.0208, 1.8945],
			up=[-0.2779, -0.9482, 0.1556]
		)

	def preprocess_point_cloud(pcd, voxel_size):
		print(":: Downsample with a voxel size %.3f." % voxel_size)
		pcd_down = pcd.voxel_down_sample(voxel_size)

		radius_normal = voxel_size * 2
		print(":: Estimate normal with search radius %.3f." % radius_normal)
		pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

		radius_feature = voxel_size * 5
		print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
		pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
			pcd_down,
			o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
		)
		return pcd_down, pcd_fpfh

	def prepare_dataset(voxel_size):
		print(":: Load two point clouds and disturb initial pose.")
		source = o3d.io.read_point_cloud("./test_data/ICP/cloud_bin_0.pcd")
		target = o3d.io.read_point_cloud("./test_data/ICP/cloud_bin_1.pcd")

		print("#source points = {}.".format(len(source.points)))
		print("#target points = {}.".format(len(target.points)))

		trans_init = np.asarray([
			[0.0, 0.0, 1.0, 0.0],
			[1.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0]
		])
		source.transform(trans_init)
		draw_registration_result(source, target, np.identity(4))

		source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
		target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
		return source, target, source_down, target_down, source_fpfh, target_fpfh

	voxel_size = 0.05  # Means 5cm for this dataset.
	source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

	# Global registration.
	if False:
		# RANSAC.
		def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
			distance_threshold = voxel_size * 1.5
			print(":: RANSAC registration on downsampled point clouds.")
			print("   Since the downsampling voxel size is %.3f," % voxel_size)
			print("   we use a liberal distance threshold %.3f." % distance_threshold)
			result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
				source_down, target_down, source_fpfh, target_fpfh, True,
				distance_threshold,
				o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
				3, [
					o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
					o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
				], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
			)
			return result

		start_time = time.time()
		result_ransac = execute_global_registration(
			source_down, target_down,
			source_fpfh, target_fpfh,
			voxel_size
		)
		print("RANSA applied: {} secs.".format(time.time() - start_time))
		print(result_ransac)
		draw_registration_result(source_down, target_down, result_ransac.transformation)

		result_global = result_ransac
	else:
		# Fast global registration (FGR).
		def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
			distance_threshold = voxel_size * 0.5
			print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
			# FPFH-feature-based FGR.
			result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
				source_down, target_down, source_fpfh, target_fpfh,
				o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
			)
			return result
		def execute_fast_global_registration_based_on_correspondence(source_down, target_down, correspondence_set):
			# Correspondence-based FGR.
			result = o3d.pipelines.registration.registration_fgr_based_on_correspondence(
				source_down, target_down, correspondence_set,
				o3d.pipelines.registration.FastGlobalRegistrationOption()
			)
			return result

		start_time = time.time()
		result_fast = execute_fast_global_registration(
			source_down, target_down,
			source_fpfh, target_fpfh,
			voxel_size
		)
		print("Fast global registration applied: {} secs.".format(time.time() - start_time))
		print(result_fast)
		draw_registration_result(source_down, target_down, result_fast.transformation)

		result_global = result_fast

	# Local refinement.
	def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
		distance_threshold = voxel_size * 0.4
		print(":: Point-to-plane ICP registration is applied on original point")
		print("   clouds to refine the alignment. This time we use a strict")
		print("   distance threshold %.3f." % distance_threshold)
		result = o3d.pipelines.registration.registration_icp(
			source, target, distance_threshold, result_global.transformation,
			o3d.pipelines.registration.TransformationEstimationPointToPlane()
		)
		return result

	start_time = time.time()
	result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
	print("Local refinement applied: {} secs.".format(time.time() - start_time))
	print(result_icp)
	draw_registration_result(source, target, result_icp.transformation)

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html
def multiway_registration_tutorial():
	def load_point_clouds(voxel_size=0.0):
		pcds = []
		for i in range(3):
			pcd = o3d.io.read_point_cloud("./test_data/ICP/cloud_bin_%d.pcd" % i)
			pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
			pcds.append(pcd_down)
		return pcds

	voxel_size = 0.02
	pcds_down = load_point_clouds(voxel_size)
	o3d.visualization.draw_geometries(
		pcds_down,
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

	# Pose graph.
	def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):
		print("Apply point-to-plane ICP")
		icp_coarse = o3d.pipelines.registration.registration_icp(
			source, target, max_correspondence_distance_coarse,
			np.identity(4),
			o3d.pipelines.registration.TransformationEstimationPointToPlane()
		)
		icp_fine = o3d.pipelines.registration.registration_icp(
			source, target, max_correspondence_distance_fine,
			icp_coarse.transformation,
			o3d.pipelines.registration.TransformationEstimationPointToPlane()
		)
		transformation_icp = icp_fine.transformation
		information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
			source, target, max_correspondence_distance_fine,
			icp_fine.transformation
		)
		return transformation_icp, information_icp

	def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
		pose_graph = o3d.pipelines.registration.PoseGraph()
		odometry = np.identity(4)
		pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
		n_pcds = len(pcds)
		for source_id in range(n_pcds):
			for target_id in range(source_id + 1, n_pcds):
				transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
				print("Build o3d.pipelines.registration.PoseGraph")
				if target_id == source_id + 1:  # Odometry case.
					odometry = np.dot(transformation_icp, odometry)
					pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
					pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=False))
				else:  # Loop closure case.
					pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp, information_icp, uncertain=True))
		return pose_graph

	print("Full registration ...")
	max_correspondence_distance_coarse = voxel_size * 15
	max_correspondence_distance_fine = voxel_size * 1.5
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		pose_graph = full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)

	print("Optimizing PoseGraph ...")
	option = o3d.pipelines.registration.GlobalOptimizationOption(
		max_correspondence_distance=max_correspondence_distance_fine,
		edge_prune_threshold=0.25,
		reference_node=0
	)
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		# The global optimization performs twice on the pose graph.
		# The first pass optimizes poses for the original pose graph taking all edges into account and does its best to distinguish false alignments among uncertain edges.
		# These false alignments have small line process weights, and they are pruned after the first pass.
		# The second pass runs without them and produces a tight global alignment.
		o3d.pipelines.registration.global_optimization(
			pose_graph,
			o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
			o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
			option
		)

	# Visualize optimization.
	print("Transform points and display")
	for point_id in range(len(pcds_down)):
		print(pose_graph.nodes[point_id].pose)
		pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
	o3d.visualization.draw_geometries(
		pcds_down,
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

	# Make a combined point cloud.
	pcds = load_point_clouds(voxel_size)
	pcd_combined = o3d.geometry.PointCloud()
	for point_id in range(len(pcds)):
		pcds[point_id].transform(pose_graph.nodes[point_id].pose)
		pcd_combined += pcds[point_id]
	pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
	o3d.io.write_point_cloud("./multiway_registration.pcd", pcd_combined_down)
	o3d.visualization.draw_geometries(
		[pcd_combined_down],
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

def main():
	#icp_registration_tutorial()
	#colored_point_cloud_registration_tutorial()
	t_icp_registration_tutorial()

	#global_registration_tutorial()
	#multiway_registration_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
