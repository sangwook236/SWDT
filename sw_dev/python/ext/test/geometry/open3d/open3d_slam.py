#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import open3d as o3d

# REF [site] >> http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html
def rgbd_integration_tutorial():
	# Read trajectory from .log file.
	class CameraPose:
		def __init__(self, meta, mat):
			self.metadata = meta
			self.pose = mat

		def __str__(self):
			return "Metadata : " + " ".join(map(str, self.metadata)) + "\n" + "Pose : " + "\n" + np.array_str(self.pose)

	def read_trajectory(filename):
		traj = []
		with open(filename, "r") as f:
			metastr = f.readline()
			while metastr:
				metadata = list(map(int, metastr.split()))
				mat = np.zeros(shape=(4, 4))
				for i in range(4):
					matstr = f.readline()
					mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
				traj.append(CameraPose(metadata, mat))
				metastr = f.readline()
		return traj

	redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
	camera_poses = read_trajectory(redwood_rgbd.odometry_log_path)

	# TSDF volume integration.
	volume = o3d.pipelines.integration.ScalableTSDFVolume(
		voxel_length=4.0 / 512.0,
		sdf_trunc=0.04,
		color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
	)

	for i in range(len(camera_poses)):
		print("Integrate {:d}-th image into the volume.".format(i))
		color = o3d.io.read_image(redwood_rgbd.color_paths[i])
		depth = o3d.io.read_image(redwood_rgbd.depth_paths[i])
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
		volume.integrate(
			rgbd,
			o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
			np.linalg.inv(camera_poses[i].pose)
		)

	# Extract a mesh.
	print("Extract a triangle mesh from the volume and visualize it.")
	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	o3d.visualization.draw_geometries(
		[mesh],
		front=[0.5297, -0.1873, -0.8272],
		lookat=[2.0712, 2.0312, 1.7251],
		up=[-0.0558, -0.9809, 0.1864],
		zoom=0.47
	)

# REF [site] >>
#	http://www.open3d.org/docs/release/tutorial/pipelines/color_map_optimization.html
#	https://github.com/isl-org/Open3D/blob/master/examples/python/reconstruction_system/color_map_optimization_for_reconstruction_system.py
def color_map_optimization_tutorial():
	def load_fountain_dataset():
		rgbd_images = []
		fountain_rgbd_dataset = o3d.data.SampleFountainRGBDImages()
		for i in range(len(fountain_rgbd_dataset.depth_paths)):
			depth = o3d.io.read_image(fountain_rgbd_dataset.depth_paths[i])
			color = o3d.io.read_image(fountain_rgbd_dataset.color_paths[i])
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
			rgbd_images.append(rgbd_image)

		camera_trajectory = o3d.io.read_pinhole_camera_trajectory(fountain_rgbd_dataset.keyframe_poses_log_path)
		mesh = o3d.io.read_triangle_mesh(fountain_rgbd_dataset.reconstruction_path)

		return mesh, rgbd_images, camera_trajectory

	# Load dataset.
	mesh, rgbd_images, camera_trajectory = load_fountain_dataset()

	# Before full optimization, let's visualize texture map with given geometry, RGBD images, and camera poses.
	mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
		mesh, rgbd_images, camera_trajectory,
		o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0)
	)
	o3d.visualization.draw_geometries(
		[mesh],
		zoom=0.5399,
		front=[0.0665, -0.1107, -0.9916],
		lookat=[0.7353, 0.6537, 1.0521],
		up=[0.0136, -0.9936, 0.1118]
	)

	# Rigid Optimization.
	# Optimize texture and save the mesh as texture_mapped.ply.
	# "Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras", Q.-Y. Zhou and V. Koltun, SIGGRAPH 2014.

	is_ci = True

	# Run rigid optimization.
	maximum_iteration = 100 if is_ci else 300
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		mesh, camera_trajectory = o3d.pipelines.color_map.run_rigid_optimizer(
			mesh, rgbd_images, camera_trajectory,
			o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=maximum_iteration)
		)

	o3d.visualization.draw_geometries(
		[mesh],
		zoom=0.5399,
		front=[0.0665, -0.1107, -0.9916],
		lookat=[0.7353, 0.6537, 1.0521],
		up=[0.0136, -0.9936, 0.1118]
	)

	# Non-rigid Optimization.
	# Run non-rigid optimization.
	maximum_iteration = 100 if is_ci else 300
	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		mesh, camera_trajectory = o3d.pipelines.color_map.run_non_rigid_optimizer(
			mesh, rgbd_images, camera_trajectory,
			o3d.pipelines.color_map.NonRigidOptimizerOption(maximum_iteration=maximum_iteration)
		)

	o3d.visualization.draw_geometries(
		[mesh],
		zoom=0.5399,
		front=[0.0665, -0.1107, -0.9916],
		lookat=[0.7353, 0.6537, 1.0521],
		up=[0.0136, -0.9936, 0.1118]
	)

def main():
	#rgbd_integration_tutorial()
	#color_map_optimization_tutorial()

	#-----
	# Pose graph optimization (PGO).
	#	Refer to pose_graph_optimization_tensor_tutorial() in ./open3d_reconstruction.py.

	# Dense RGB-D SLAM.
	#	Refer to dense_slam_tensor_tutorial() in ./open3d_reconstruction.py.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
