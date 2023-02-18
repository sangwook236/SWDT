#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import open3d as o3d

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html
def rgbd_odometry_tutorial():
	# Read camera intrinsic.
	redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()

	pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(redwood_rgbd.camera_intrinsic_path)
	print(pinhole_camera_intrinsic.intrinsic_matrix)

	# Read RGBD image.
	source_color = o3d.io.read_image(redwood_rgbd.color_paths[0])
	source_depth = o3d.io.read_image(redwood_rgbd.depth_paths[0])
	target_color = o3d.io.read_image(redwood_rgbd.color_paths[1])
	target_depth = o3d.io.read_image(redwood_rgbd.depth_paths[1])
	source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
	target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
	target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)

	# Compute odometry from two RGBD image pairs.
	option = o3d.pipelines.odometry.OdometryOption()
	odo_init = np.identity(4)
	print(option)

	[success_color_term, trans_color_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
		source_rgbd_image, target_rgbd_image,
		pinhole_camera_intrinsic, odo_init,
		o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
		option
	)
	[success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
		source_rgbd_image, target_rgbd_image,
		pinhole_camera_intrinsic, odo_init,
		o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
		option
	)

	# Visualize RGBD image pairs.
	if success_color_term:
		print("Using RGB-D Odometry")
		print(trans_color_term)
		source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
		source_pcd_color_term.transform(trans_color_term)
		o3d.visualization.draw_geometries(
			[target_pcd, source_pcd_color_term],
			zoom=0.48,
			front=[0.0999, -0.1787, -0.9788],
			lookat=[0.0345, -0.0937, 1.8033],
			up=[-0.0067, -0.9838, 0.1790]
		)
	if success_hybrid_term:
		print("Using Hybrid RGB-D Odometry")
		print(trans_hybrid_term)
		source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
		source_pcd_hybrid_term.transform(trans_hybrid_term)
		o3d.visualization.draw_geometries(
			[target_pcd, source_pcd_hybrid_term],
			zoom=0.48,
			front=[0.0999, -0.1787, -0.9788],
			lookat=[0.0345, -0.0937, 1.8033],
			up=[-0.0067, -0.9838, 0.1790]
		)

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/rgbd_odometry.py
def rgbd_odometry_tensor_tutorial():
	lounge_rgbd = o3d.data.LoungeRGBDImages()

	intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	intrinsic = o3d.core.Tensor(intrinsic_legacy.intrinsic_matrix, o3d.core.Dtype.Float64)

	idx1, idx2 = 0, 10

	# Read RGBD image.
	depth_src = o3d.t.io.read_image(lounge_rgbd.depth_paths[idx1])
	color_src = o3d.t.io.read_image(lounge_rgbd.color_paths[idx1])
	depth_dst = o3d.t.io.read_image(lounge_rgbd.depth_paths[idx2])
	color_dst = o3d.t.io.read_image(lounge_rgbd.color_paths[idx2])
	rgbd_src = o3d.t.geometry.RGBDImage(color_src, depth_src)
	rgbd_dst = o3d.t.geometry.RGBDImage(color_dst, depth_dst)

	#-----
	# RGBD odometry and information matrix computation.
	start_time = time.time()
	res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
		rgbd_src, rgbd_dst,  # CPU.
		#rgbd_src.cuda(0), rgbd_dst.cuda(0),  # CUDA.
		intrinsic,
		init_source_to_target=o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32),
		#init_source_to_target=o3d.core.Tensor(np.eye(4)),
		depth_scale=1000.0, depth_max=3.0,
		criteria_list=[o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10), o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5), o3d.t.pipelines.odometry.OdometryConvergenceCriteria(3)],
		method=o3d.t.pipelines.odometry.Hybrid,  # {PointToPlane, Intensity, Hybrid}.
		params=o3d.t.pipelines.odometry.OdometryLossParams()
	)
	print("RGBD odometry done: {} secs.".format(time.time() - start_time))
	print("Inlier RMSE = {}, fitness = {}.".format(res.inlier_rmse, res.fitness))  # The overlapping area (# of inlier correspondences / # of points in target).
	print("Odometry:\n{}.".format(res.transformation))
	info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
		depth_src, depth_dst,
		intrinsic,
		source_to_target=res.transformation,
		dist_thr=0.07, depth_scale=1000.0, depth_max=3.0
	)
	print("Information matrix:\n{}.".format(info))
	print(info[5, 5] / (depth_src.columns * depth_src.rows))

	#-----
	# Legacy for reference, can be a little bit different due to minor implementation discrepancies.
	def read_legacy_rgbd_image(color_file, depth_file, convert_rgb_to_intensity):
		color = o3d.io.read_image(color_file)
		depth = o3d.io.read_image(depth_file)
		rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color, depth,
			depth_scale=1000.0, depth_trunc=3.0,
			convert_rgb_to_intensity=convert_rgb_to_intensity
		)
		return rgbd_image

	rgbd_src_legacy = read_legacy_rgbd_image(lounge_rgbd.color_paths[idx1], lounge_rgbd.depth_paths[idx1], True)
	rgbd_dst_legacy = read_legacy_rgbd_image(lounge_rgbd.color_paths[idx2], lounge_rgbd.depth_paths[idx2], True)
	start_time = time.time()
	success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
		rgbd_src_legacy, rgbd_dst_legacy,
		intrinsic_legacy, np.eye(4),
		o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
	)
	print("RGBD odometry (legacy) done: {} secs.".format(time.time() - start_time))
	print("Success = {}.".format(success))
	print("Odometry:\n{}.".format(trans))
	print("Information matrix:\n{}.".format(info))

	#-----
	# Visualization.
	pcd_src = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_src, intrinsic)
	pcd_dst = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_dst, intrinsic)
	o3d.visualization.draw([pcd_src, pcd_dst])
	o3d.visualization.draw([pcd_src.transform(res.transformation), pcd_dst])

def main():
	#rgbd_odometry_tutorial()
	rgbd_odometry_tensor_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
