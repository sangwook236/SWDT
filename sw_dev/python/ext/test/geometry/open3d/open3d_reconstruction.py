#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, warnings, glob, json, time
from dataclasses import dataclass
import numpy as np
import open3d as o3d
#import open3d_tutorial as o3dtut
#import matplotlib.pyplot as plt
#import reconstruction_system as recon_sys
from tqdm import tqdm

def reconstruction_system_tutorial():
	# REF [site] >> http://www.open3d.org/docs/release/tutorial/reconstruction_system/system_overview.html
	config = {
		"name": "Open3D reconstruction tutorial http://open3d.org/docs/release/tutorial/reconstruction_system/system_overview.html",
		#"path_dataset": "dataset/tutorial/",
		"path_dataset": "016",
		"path_intrinsic": "",
		"max_depth": 3.0,
		"voxel_size": 0.05,
		"max_depth_diff": 0.07,
		"preference_loop_closure_odometry": 0.1,
		"preference_loop_closure_registration": 5.0,
		"tsdf_cubic_size": 3.0,
		"icp_method": "color",
		"global_registration": "ransac",
		"python_multi_threading": True
	}

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/run_system.py
def pose_graph_optimization_tensor_tutorial():
	# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/rgbd_odometry.py
	def rgbd_loop_closure(depth_list, color_list, intrinsic, device):
		odometry_loop_interval = 10
		n_files = len(depth_list)

		key_indices = list(range(0, n_files, odometry_loop_interval))
		n_key_indices = len(key_indices)

		criteria_list = [
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5)
		]
		method = o3d.t.pipelines.odometry.Method.PointToPlane

		edges, poses, infos = [], [], []
		for i in range(n_key_indices - 1):
			key_i = key_indices[i]
			depth_curr = o3d.t.io.read_image(depth_list[key_i]).to(device)
			color_curr = o3d.t.io.read_image(color_list[key_i]).to(device)
			rgbd_curr = o3d.t.geometry.RGBDImage(color_curr, depth_curr)

			for j in range(i + 1, n_key_indices):
				key_j = key_indices[j]
				depth_next = o3d.t.io.read_image(depth_list[key_j]).to(device)
				color_next = o3d.t.io.read_image(color_list[key_j]).to(device)
				rgbd_next = o3d.t.geometry.RGBDImage(color_next, depth_next)

				# TODO: Add OpenCV initialization if necessary.
				# TODO: Better failure check.
				try:
					res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
						rgbd_curr, rgbd_next,
						intrinsic, o3d.core.Tensor(np.eye(4)),
						1000.0, 3.0,
						criteria_list, method
					)
					info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
						depth_curr, depth_next,
						intrinsic, res.transformation,
						0.07, 1000.0, 3.0
					)
				except Exception as e:
					pass
				else:
					if info[5, 5] / (depth_curr.columns * depth_curr.rows) > 0.3:
						edges.append((key_i, key_j))
						poses.append(res.transformation.cpu().numpy())
						infos.append(info.cpu().numpy())

						#pcd_src = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_curr, intrinsic)
						#pcd_dst = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_next, intrinsic)
						#o3d.visualization.draw([pcd_src, pcd_dst])
						#o3d.visualization.draw([pcd_src.transform(res.transformation), pcd_dst])

		return edges, poses, infos

	# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/rgbd_odometry.py
	def rgbd_odometry(depth_list, color_list, intrinsic, device):
		n_files = len(depth_list)

		depth_curr = o3d.t.io.read_image(depth_list[0]).to(device)
		color_curr = o3d.t.io.read_image(color_list[0]).to(device)
		rgbd_curr = o3d.t.geometry.RGBDImage(color_curr, depth_curr)

		criteria_list = [
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
			o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5)
		]
		method = o3d.t.pipelines.odometry.Method.PointToPlane

		edges, poses, infos = [], [], []
		for i in tqdm(range(0, n_files - 1)):
			depth_next = o3d.t.io.read_image(depth_list[i + 1]).to(device)
			color_next = o3d.t.io.read_image(color_list[i + 1]).to(device)
			rgbd_next = o3d.t.geometry.RGBDImage(color_next, depth_next)

			res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
				rgbd_curr, rgbd_next,
				intrinsic, o3d.core.Tensor(np.eye(4)),
				1000.0, 3.0,
				criteria_list, method
			)
			# FIXME [restore] >>
			"""
			info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
				depth_curr, depth_next,
				intrinsic, res.transformation,
				0.07, 1000.0, 3.0
			)
			"""
			info = o3d.core.Tensor(np.eye(6)).to(device)

			edges.append((i, i + 1))
			poses.append(res.transformation.cpu().numpy())
			infos.append(info.cpu().numpy())

			color_curr = color_next
			depth_curr = depth_next
			rgbd_curr = rgbd_next

		return edges, poses, infos

	# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/integrate.py
	def integrate(vbg, depth_list, color_list, depth_intrinsic, color_intrinsic, extrinsics, integrate_color, device):
		start_time = time.time()
		n_files = len(depth_list)
		for i in tqdm(range(n_files)):
			depth = o3d.t.io.read_image(depth_list[i]).to(device)
			extrinsic = extrinsics[i]

			frustum_block_coords = vbg.compute_unique_block_coordinates(
				depth,
				depth_intrinsic, extrinsic,
				depth_scale, depth_max
			)

			if integrate_color:
				color = o3d.t.io.read_image(color_list[i]).to(device)
				vbg.integrate(
					frustum_block_coords,
					depth, color,
					depth_intrinsic, color_intrinsic, extrinsic,
					depth_scale, depth_max
				)
			else:
				vbg.integrate(
					frustum_block_coords,
					depth,
					depth_intrinsic, extrinsic,
					depth_scale, depth_max
				)
			dt = time.time() - start_time
		print("Finished integrating {} frames in {} seconds.".format(n_files, dt))

	#-----
	lounge_rgbd = o3d.data.LoungeRGBDImages()

	intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	intrinsic = o3d.core.Tensor(intrinsic_legacy.intrinsic_matrix, o3d.core.Dtype.Float64)

	# Read RGBD image.
	depth_list = lounge_rgbd.depth_paths
	color_list = lounge_rgbd.color_paths
	assert len(depth_list) == len(color_list)

	device = o3d.core.Device("CUDA:0")

	#-----
	# Split into fragments.

	fragment_size = 100
	fragment_overlap_size = 1
	depth_lists, color_lists = [], []
	for idx in range(0, len(depth_list), fragment_size):
		#depth_lists.append(depth_list[idx:idx + fragment_size])
		#color_lists.append(color_list[idx:idx + fragment_size])
		# One-frame overlap between consecutive fragments for inter-fragment registration.
		depth_lists.append(depth_list[idx:idx + fragment_size + fragment_overlap_size])
		color_lists.append(color_list[idx:idx + fragment_size + fragment_overlap_size])

	#-----
	# Odometry in each fragment.

	method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
	criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
	option = o3d.pipelines.registration.GlobalOptimizationOption(
		max_correspondence_distance=0.07,
		edge_prune_threshold=0.25,
		preference_loop_closure=0.1,
		reference_node=0
	)

	start = time.time()
	pose_graphs = []
	for frag_id, (depth_list, color_list) in enumerate(zip(depth_lists, color_lists)):
		pose_graph = o3d.pipelines.registration.PoseGraph()

		# Odometry: (i, i + 1), trans(i, i + 1), info(i, i + 1).
		edges, trans, infos = rgbd_odometry(depth_list, color_list, intrinsic, device)
		pose_i2w = np.eye(4)
		pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose_i2w.copy()))

		edges_for_check = []
		for i in range(len(trans)):
			trans_i2j = trans[i]
			info_i2j = infos[i]

			trans_j2i = np.linalg.inv(trans_i2j)
			pose_j2w = pose_i2w @ trans_j2i  # T_wj = T_wi * T_ij.

			pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose_j2w.copy()))
			assert len(pose_graph.nodes) == i + 2
			pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i + 1, trans_i2j.copy(), info_i2j.copy(), uncertain=False, confidence=1.0))
			edges_for_check.append((i, i + 1))
			pose_i2w = pose_j2w

		# Loop closure: (i, j), trans(i, j), info(i, j) where i, j are multipliers of intervals.
		edges, trans, infos = rgbd_loop_closure(depth_list, color_list, intrinsic, device)
		for i in range(len(edges)):
			if edges[i] in edges_for_check:
				print("Warning: edge ({}, {}) already exists.".format(*edges[i]))
				continue
			else:
				edges_for_check.append(edges[i])

			ki, kj = edges[i]
			trans_i2j = trans[i]
			info_i2j = infos[i]

			pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(ki, kj, trans_i2j.copy(), info_i2j.copy(), uncertain=True, confidence=1.0))

		# In-place optimization.
		with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
			o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)

		#o3d.io.write_pose_graph(os.path.join("./fragment_posegraph_{:03d}.json".format(frag_id)), pose_graph)
		pose_graphs.append(pose_graph)
	end = time.time()
	print("Pose graph generation and optimization takes {:.3f}s for {} fragments.".format(end - start, len(depth_lists)))

	#-----
	# Integrate fragmemts.

	start = time.time()
	integrate_color = True
	depth_scale = 1000.0
	depth_max = 3.0
	T_f2w = np.eye(4)

	if integrate_color:
		vbg = o3d.t.geometry.VoxelBlockGrid(
			attr_names=("tsdf", "weight", "color"),
			attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
			attr_channels=((1), (1), (3)),
			voxel_size=3.0 / 512,
			block_resolution=16,
			block_count=50000,
			device=device
		)
	else:
		vbg = o3d.t.geometry.VoxelBlockGrid(
			attr_names=("tsdf", "weight"),
			attr_dtypes=(o3d.core.float32, o3d.core.float32),
			attr_channels=((1), (1)),
			voxel_size=3.0 / 512,
			block_resolution=16,
			block_count=50000,
			device=device
		)

	for frag_id, (depth_list, color_list) in enumerate(zip(depth_lists, color_lists)):
		depth_list, color_list = depth_list[:fragment_size], color_list[:fragment_size]

		pose_graph = pose_graphs[frag_id]
		#extrinsics = [np.linalg.inv(node.pose) for node in pose_graph.nodes]
		# One-frame overlap between consecutive fragments for inter-fragment registration.
		extrinsics = [np.linalg.inv(T_f2w @ node.pose) for node in pose_graph.nodes]

		integrate(vbg, depth_list, color_list, intrinsic, intrinsic, extrinsics, integrate_color, device)

		# One-frame overlap between consecutive fragments for inter-fragment registration.
		if frag_id < len(pose_graphs) - 1:
			T_f2w = T_f2w @ pose_graph.nodes[fragment_size].pose  # T_wfj = T_wfi * T_fifj.

		# Float color does not load correctly in the o3d.t.io mode.
		#pcd = vbg.extract_point_cloud(weight_threshold=3.0)
		#o3d.io.write_point_cloud(os.path.join("./fragment_pcd_{:03d}.ply".format(frag_id)), pcd.to_legacy())
	end = time.time()
	print("TSDF integration takes {:.3f}s for {} fragments.".format(end - start, len(depth_lists)))

	# Visualize.
	if True:
		pcd = vbg.extract_point_cloud(weight_threshold=3.0).cpu()
		o3d.visualization.draw([pcd])
		#o3d.visualization.draw_geometries([pcd.to_legacy()])  # Fast.
		#o3d.io.write_point_cloud("./fragments_pcd.ply", pcd.to_legacy())
	else:
		mesh = vbg.extract_triangle_mesh(weight_threshold=3.0)
		o3d.visualization.draw([mesh.to_legacy()])
		#o3d.visualization.draw_geometries([mesh.to_legacy()])  # Fast.
		#o3d.io.write_triangle_mesh("./fragments_mesh.ply", mesh.to_legacy())

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/dense_slam.py
def dense_slam_tensor_tutorial():
	def slam(depth_file_names, color_file_names, intrinsic, config):
		n_files = len(color_file_names)
		device = o3d.core.Device(config.device)

		T_frame_to_model = o3d.core.Tensor(np.identity(4))
		model = o3d.t.pipelines.slam.Model(config.voxel_size, 16, config.block_count, T_frame_to_model, device)
		depth_ref = o3d.t.io.read_image(depth_file_names[0])
		input_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsic, device)
		raycast_frame = o3d.t.pipelines.slam.Frame(depth_ref.rows, depth_ref.columns, intrinsic, device)

		poses = []
		for i in range(n_files):
			start = time.time()

			depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
			color = o3d.t.io.read_image(color_file_names[i]).to(device)

			input_frame.set_data_from_image("depth", depth)
			input_frame.set_data_from_image("color", color)

			if i > 0:
				result = model.track_frame_to_model(
					input_frame, raycast_frame,
					depth_scale=config.depth_scale, depth_max=config.depth_max, depth_diff=config.odometry_distance_thr,
					method=o3d.t.pipelines.odometry.Method.PointToPlane,
					criteria=[o3d.t.pipelines.odometry.OdometryConvergenceCriteria(6), o3d.t.pipelines.odometry.OdometryConvergenceCriteria(3), o3d.t.pipelines.odometry.OdometryConvergenceCriteria(1)]
				)
				T_frame_to_model = T_frame_to_model @ result.transformation

			poses.append(T_frame_to_model.cpu().numpy())
			model.update_frame_pose(i, T_frame_to_model)
			model.integrate(input_frame, config.depth_scale, config.depth_max, config.trunc_voxel_multiplier)
			model.synthesize_model_frame(
				raycast_frame,
				config.depth_scale, config.depth_min, config.depth_max,
				config.trunc_voxel_multiplier,
				enable_color=False, weight_threshold=-1.0
			)

			stop = time.time()
			print("{:04d}/{:04d} SLAM takes {:.4}s.".format(i, n_files, stop - start))

		return model.voxel_grid, poses

	# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
	def save_poses(path_trajectory, poses, intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)):
		if path_trajectory.endswith("log"):
			traj = o3d.camera.PinholeCameraTrajectory()
			params = []
			for pose in poses:
				param = o3d.camera.PinholeCameraParameters()
				param.intrinsic = intrinsic
				param.extrinsic = np.linalg.inv(pose)
				params.append(param)
			traj.parameters = params
			o3d.io.write_pinhole_camera_trajectory(path_trajectory, traj)
		elif path_trajectory.endswith("json"):
			pose_graph = o3d.pipelines.registration.PoseGraph()
			for pose in poses:
				node = o3d.pipelines.registration.PoseGraphNode()
				node.pose = pose
				pose_graph.nodes.append(node)
			o3d.io.write_pose_graph(path_trajectory, pose_graph)

	# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
	def extract_trianglemesh(volume, config, file_name=None):
		if config.engine == "legacy":
			mesh = volume.extract_triangle_mesh()
			mesh.compute_vertex_normals()
			mesh.compute_triangle_normals()
			if file_name is not None:
				o3d.io.write_triangle_mesh(file_name, mesh)
		elif config.engine == "tensor":
			mesh = volume.extract_triangle_mesh(weight_threshold=config.surface_weight_thr)
			mesh = mesh.to_legacy()
			if file_name is not None:
				o3d.io.write_triangle_mesh(file_name, mesh)
		return mesh

	if False:
		# Extract RGB-D frames and intrinsic from bag file.
		# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
		def extract_rgbd_frames(rgbd_video_file):
			"""
			Extract color and aligned depth frames and intrinsic calibration from an RGBD video file (currently only RealSense bag files supported).
			Folder structure is:
				<directory of rgbd_video_file>/<rgbd_video_file name without extension>/{depth/00000.jpg, color/00000.png, intrinsic.json}
			"""
			frames_folder = os.path.join(os.path.dirname(rgbd_video_file), os.path.basename(os.path.splitext(rgbd_video_file)[0]))
			path_intrinsic = os.path.join(frames_folder, "intrinsic.json")
			if os.path.isfile(path_intrinsic):
				warnings.warn(f"Skipping frame extraction for {rgbd_video_file} since files are present.")
			else:
				rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
				rgbd_video.save_frames(frames_folder)
			with open(path_intrinsic) as intr_file:
				intr = json.load(intr_file)
			depth_scale = intr["depth_scale"]
			return frames_folder, path_intrinsic, depth_scale

		# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
		def load_depth_file_names(config):
			if not os.path.exists(config.path_dataset):
				print(
					"Path '{}' not found.".format(config.path_dataset),
					"Please provide --path_dataset in the command line or the config file."
				)
				return []

			depth_folder = os.path.join(config.path_dataset, config.depth_folder)

			# Only 16-bit png depth is supported.
			depth_file_names = glob.glob(os.path.join(depth_folder, "*.png"))
			n_depth = len(depth_file_names)
			if n_depth == 0:
				print("Depth image not found in {}, abort!".format(depth_folder))
				return []

			return sorted(depth_file_names)

		# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
		def load_rgbd_file_names(config):
			depth_file_names = load_depth_file_names(config)
			if len(depth_file_names) == 0:
				return [], []

			color_folder = os.path.join(config.path_dataset, config.color_folder)
			extensions = ["*.png", "*.jpg"]
			for ext in extensions:
				color_file_names = glob.glob(os.path.join(color_folder, ext))
				if len(color_file_names) == len(depth_file_names):
					return depth_file_names, sorted(color_file_names)

			depth_folder = os.path.join(config.path_dataset, config.depth_folder)
			print(
				"Found {} depth images in {}, but cannot find matched number of "
				"color images in {} with extensions {}, abort!".format(len(depth_file_names), depth_folder, color_folder, extensions)
			)
			return [], []

		# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/common.py
		def load_intrinsic(config, key="depth"):
			path_intrinsic = config.path_color_intrinsic if key == "color" else config.path_intrinsic

			if path_intrinsic is None or path_intrinsic == "":
				intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
			else:
				intrinsic = o3d.io.read_pinhole_camera_intrinsic(path_intrinsic)

			if config.engine == "legacy":
				return intrinsic
			elif config.engine == "tensor":
				return o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
			else:
				print("Unsupported engine {}.".format(config.engine))

		#-----
		@dataclass
		class Config:
			name: str = "Default reconstruction system config"
			device: str = "CUDA:0"  # {"CPU:0", "CUDA:0", ...}.
			engine: str = "tensor"  # {legacy, tensor}.
			multiprocessing: bool = False
			path_dataset: str = ""
			path_trajectory: str = ""
			depth_folder: str = "depth"
			color_folder: str = "color"
			path_intrinsic: str = ""
			path_color_intrinsic: str = ""
			fragment_size: int = 100
			depth_min: float = 0.1
			depth_max: float = 3.0
			depth_scale: float = 1000.0
			odometry_method: str = "hybrid"
			odometry_loop_interval: int = 10
			odometry_loop_weight: float = 0.1
			odometry_distance_thr: float = 0.07
			icp_method: str = "colored"
			icp_voxelsize: float = 0.05
			icp_distance_thr: float = 0.07
			global_registration_method: str = "ransac"
			registration_loop_weight: float = 0.1
			integration_mode: str = "color"
			voxel_size: float = 0.0058
			trunc_voxel_multiplier: float = 8.0
			block_count: int = 40000
			est_point_count: int = 6000000
			surface_weight_thr: float = 3.0
			default_dataset: str = "lounge"  # {lounge, bedroom, jack_jack}.
			path_npz: str = "./dense_slam.npz"  # Path to the npz file that stores voxel block grid.

		config = Config()

		if config.default_dataset == "lounge":
			lounge_rgbd = o3d.data.LoungeRGBDImages()
			# Override default config parameters with dataset specific parameters.
			config.path_dataset = lounge_rgbd.extract_dir
			config.path_trajectory = lounge_rgbd.trajectory_log_path
			config.depth_folder = "depth"
			config.color_folder = "color"
		elif config.default_dataset == "bedroom":
			bedroom_rgbd = o3d.data.BedroomRGBDImages()
			# Override default config parameters with dataset specific parameters.
			config.path_dataset = bedroom_rgbd.extract_dir
			config.path_trajectory = bedroom_rgbd.trajectory_log_path
			config.depth_folder = "depth"
			config.color_folder = "image"
		elif config.default_dataset == "jack_jack":
			jackjack_rgbd = o3d.data.JackJackL515Bag()
			# Override default config parameters with dataset specific parameters.
			print("Extracting frames from RGBD video file.")
			config.path_dataset = jackjack_rgbd.path
			config.depth_folder = "depth"
			config.color_folder = "color"
		else:
			print("The requested dataset is not available. Available dataset options include lounge and jack_jack.")
			return

		if config.path_dataset.endswith(".bag"):
			assert os.path.isfile(config.path_dataset), (f"File {config.path_dataset} not found.")
			print("Extracting frames from RGBD video file.")
			config.path_dataset, config.path_intrinsic, config.depth_scale = extract_rgbd_frames(config.path_dataset)

		depth_file_names, color_file_names = load_rgbd_file_names(config)
		intrinsic = load_intrinsic(config)
	else:
		@dataclass
		class Config:
			name: str = "Default reconstruction system config"
			device: str = "CUDA:0"  # {"CPU:0", "CUDA:0", ...}.
			engine: str = "tensor"  # {legacy, tensor}.
			multiprocessing: bool = False
			depth_min: float = 0.1
			depth_max: float = 3.0
			depth_scale: float = 1000.0
			odometry_distance_thr: float = 0.07
			voxel_size: float = 0.0058
			trunc_voxel_multiplier: float = 8.0
			block_count: int = 40000
			surface_weight_thr: float = 3.0
			path_npz: str = "./dense_slam.npz"  # Path to the npz file that stores voxel block grid.

		config = Config()

		#-----
		lounge_rgbd = o3d.data.LoungeRGBDImages()

		intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
		intrinsic = o3d.core.Tensor(intrinsic_legacy.intrinsic_matrix, o3d.core.Dtype.Float64)

		# Read RGBD image.
		depth_file_names = lounge_rgbd.depth_paths
		color_file_names = lounge_rgbd.color_paths

	#-----
	if not os.path.exists(config.path_npz):
		start_time = time.time()
		volume, poses = slam(depth_file_names, color_file_names, intrinsic, config)
		print("Dense SLAM takes {:.4}s.".format(time.time() - start_time))

		print("Saving to {}...".format(config.path_npz))
		volume.save(config.path_npz)
		save_poses("./dense_slam_pose_output.log", poses)
		print("Saving finished.")
	else:
		print("Loading from {}...".format(config.path_npz))
		volume = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)

	mesh = extract_trianglemesh(volume, config, "./dense_slam_output.ply")
	o3d.visualization.draw([mesh])

def main():
	# Reconstruction system.
	#	REF [site] >>
	#		http://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html
	#		https://github.com/isl-org/Open3D/tree/master/examples/python/reconstruction_system
	#
	# Procedure.
	#	Make fragments.
	#	Register fragments.
	#	Refine registration.
	#	Integrate scene.

	#reconstruction_system_tutorial()  # Not yet completed.

	#-----
	# Reconstruction system (Tensor).
	#	REF [site] >>
	#		http://www.open3d.org/docs/release/tutorial/t_reconstruction_system/index.html
	#		https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system

	# Pose graph optimization (PGO).
	#pose_graph_optimization_tensor_tutorial()  # Error in inter-fragment registration.

	# Dense RGB-D SLAM.
	dense_slam_tensor_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
