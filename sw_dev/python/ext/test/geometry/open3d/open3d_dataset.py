#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import numpy as np
import open3d as o3d
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def visualize_rgbd(rgbd_image):
	print(rgbd_image)

	o3d.visualization.draw_geometries([rgbd_image])

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	)
	# Flip it, otherwise the pointcloud will be upside down.
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd])

# This is special function used for reading NYU pgm format as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder=">"):
	with open(filename, "rb") as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer
		).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	img = np.frombuffer(
		buffer,
		dtype=byteorder + "u2",
		count=int(width) * int(height),
		offset=len(header)).reshape((int(height), int(width))
	)
	img_out = img.astype("u2")
	return img_out

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/geometry/rgbd_datasets.py
def nyu_dataset_example():
	print("Read NYU dataset")
	# Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
	# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
	nyu_data = o3d.data.SampleNYURGBDImage()
	#	SampleNYURGBDImage.color_path.
	#	SampleNYURGBDImage.depth_path.
	#	Dataset.data_root.
	#	Dataset.prefix.
	#	Dataset.download_dir.
	#	Dataset.extract_dir.
	color_raw = mpimg.imread(nyu_data.color_path)
	depth_raw = read_nyu_pgm(nyu_data.depth_path)
	color = o3d.geometry.Image(color_raw)
	depth = o3d.geometry.Image(depth_raw)
	rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(
		color, depth, convert_rgb_to_intensity=False
	)

	print("Displaying NYU color and depth images and pointcloud ...")
	visualize_rgbd(rgbd_image)

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/geometry/rgbd_datasets.py
def redwood_dataset_example():
	print("Read Redwood dataset")
	redwood_data = o3d.data.SampleRedwoodRGBDImages()
	#	SampleRedwoodRGBDImages.color_paths.
	#	SampleRedwoodRGBDImages.depth_paths.
	#	SampleRedwoodRGBDImages.rgbd_match_path.
	#	SampleRedwoodRGBDImages.reconstruction_path.
	#	SampleRedwoodRGBDImages.trajectory_log_path.
	#	SampleRedwoodRGBDImages.odometry_log_path.
	#	SampleRedwoodRGBDImages.camera_intrinsic_path.
	#	Dataset.data_root.
	#	Dataset.prefix.
	#	Dataset.download_dir.
	#	Dataset.extract_dir.
	color_raw = o3d.io.read_image(redwood_data.color_paths[0])
	depth_raw = o3d.io.read_image(redwood_data.depth_paths[0])
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_raw, depth_raw, convert_rgb_to_intensity=False
	)

	print("Displaying Redwood color and depth images and pointcloud ...")
	visualize_rgbd(rgbd_image)

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/geometry/rgbd_datasets.py
def sun_dataset_example():
	print("Read SUN dataset")
	sun_data = o3d.data.SampleSUNRGBDImage()
	#	SampleSUNRGBDImage.color_path.
	#	SampleSUNRGBDImage.depth_path.
	#	Dataset.data_root.
	#	Dataset.prefix.
	#	Dataset.download_dir.
	#	Dataset.extract_dir.
	color_raw = o3d.io.read_image(sun_data.color_path)
	depth_raw = o3d.io.read_image(sun_data.depth_path)
	rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
		color_raw, depth_raw, convert_rgb_to_intensity=False
	)

	print("Displaying SUN color and depth images and pointcloud ...")
	visualize_rgbd(rgbd_image)

# REF [site] >> https://github.com/isl-org/Open3D/blob/master/examples/python/geometry/rgbd_datasets.py
def tum_dataset_example():
	print("Read TUM dataset")
	tum_data = o3d.data.SampleTUMRGBDImage()
	#	SampleTUMRGBDImage.color_path.
	#	SampleTUMRGBDImage.depth_path.
	#	Dataset.data_root.
	#	Dataset.prefix.
	#	Dataset.download_dir.
	#	Dataset.extract_dir.
	color_raw = o3d.io.read_image(tum_data.color_path)
	depth_raw = o3d.io.read_image(tum_data.depth_path)
	rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(
		color_raw, depth_raw, convert_rgb_to_intensity=False
	)

	print("Displaying TUM color and depth images and pointcloud ...")
	visualize_rgbd(rgbd_image)

# REF [site] >> http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
def redwood_dataset_tutorial():
	print("Read Redwood dataset")
	color_raw = o3d.io.read_image("../../test_data/RGBD/color/00000.jpg")
	depth_raw = o3d.io.read_image("../../test_data/RGBD/depth/00000.png")
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
	print(rgbd_image)

	plt.subplot(1, 2, 1)
	plt.title("Redwood grayscale image")
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title("Redwood depth image")
	plt.imshow(rgbd_image.depth)
	plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	)
	# Flip it, otherwise the pointcloud will be upside down.
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd], zoom=0.5)

# REF [site] >> http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
def sun_dataset_tutorial():
	print("Read SUN dataset")
	color_raw = o3d.io.read_image("../../test_data/RGBD/other_formats/SUN_color.jpg")
	depth_raw = o3d.io.read_image("../../test_data/RGBD/other_formats/SUN_depth.png")
	rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)
	print(rgbd_image)

	plt.subplot(1, 2, 1)
	plt.title("SUN grayscale image")
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title("SUN depth image")
	plt.imshow(rgbd_image.depth)
	plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	)
	# Flip it, otherwise the pointcloud will be upside down.
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd], zoom=0.5)

# REF [site] >> http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
def nyu_dataset_tutorial():
	print("Read NYU dataset")
	# Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
	# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
	color_raw = mpimg.imread("../../test_data/RGBD/other_formats/NYU_color.ppm")
	depth_raw = read_nyu_pgm("../../test_data/RGBD/other_formats/NYU_depth.pgm")
	color = o3d.geometry.Image(color_raw)
	depth = o3d.geometry.Image(depth_raw)
	rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
	print(rgbd_image)

	plt.subplot(1, 2, 1)
	plt.title("NYU grayscale image")
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title("NYU depth image")
	plt.imshow(rgbd_image.depth)
	plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	)
	# Flip it, otherwise the pointcloud will be upside down.
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd], zoom=0.5)

# REF [site] >> http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
def tum_dataset_tutorial():
	print("Read TUM dataset")
	color_raw = o3d.io.read_image("../../test_data/RGBD/other_formats/TUM_color.png")
	depth_raw = o3d.io.read_image("../../test_data/RGBD/other_formats/TUM_depth.png")
	rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
	print(rgbd_image)

	plt.subplot(1, 2, 1)
	plt.title("TUM grayscale image")
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title("TUM depth image")
	plt.imshow(rgbd_image.depth)
	plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
	)
	# Flip it, otherwise the pointcloud will be upside down.
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd], zoom=0.35)

def point_cloud_tutorial():
	if True:
		dataset = o3d.data.PCDPointCloud()
		pcd = o3d.io.read_point_cloud(dataset.path)
		o3d.visualization.draw(pcd)

	if False:
		dataset = o3d.data.PLYPointCloud()
		pcd = o3d.io.read_point_cloud(dataset.path)
		o3d.visualization.draw(pcd)

	if False:
		dataset = o3d.data.PTSPointCloud()
		pcd = o3d.io.read_point_cloud(dataset.path)
		o3d.visualization.draw(pcd)

	if False:
		dataset = o3d.data.EaglePointCloud()
		pcd = o3d.io.read_point_cloud(dataset.path)
		o3d.visualization.draw(pcd)

	if False:
		dataset = o3d.data.LivingRoomPointClouds()
		pcds = []
		for pcd_path in dataset.paths:
			pcds.append(o3d.io.read_point_cloud(pcd_path))
		o3d.visualization.draw(pcds)

	if False:
		dataset = o3d.data.OfficePointClouds()
		pcds = []
		for pcd_path in dataset.paths:
			pcds.append(o3d.io.read_point_cloud(pcd_path))
		o3d.visualization.draw(pcds)

def triangle_mesh_tutorial():
	if True:
		dataset = o3d.data.ArmadilloMesh()
		mesh = o3d.io.read_triangle_mesh(dataset.path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

	if False:
		dataset = o3d.data.BunnyMesh()
		mesh = o3d.io.read_triangle_mesh(dataset.path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

	if False:
		dataset = o3d.data.KnotMesh()
		mesh = o3d.io.read_triangle_mesh(dataset.path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

def triangle_mesh_with_prb_texture_tutorial():
	if True:
		dataset = o3d.data.AvocadoModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

	if False:
		dataset = o3d.data.CrateModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

	if False:
		dataset = o3d.data.DamagedHelmetModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

	if False:
		dataset = o3d.data.FlightHelmetModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

	if False:
		dataset = o3d.data.MonkeyModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

	if False:
		dataset = o3d.data.SwordModel()
		model = o3d.io.read_triangle_model(dataset.path)
		o3d.visualization.draw(model)

def texture_material_images_tutorial():
	if True:
		mat_data = o3d.data.MetalTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)
		mat.metallic_img = o3d.io.read_image(mat_data.metallic_texture_path)

		#sphere = o3d.geometry.TriangleMesh.create_sphere(0.5)
		#sphere.compute_vertex_normals()
		scene.add_geometry("Sphere", sphere, mat)

	if False:
		mat_data = o3d.data.PaintedPlasterTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

	if False:
		mat_data = o3d.data.TerrazzoTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

	if False:
		mat_data = o3d.data.TilesTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

	if False:
		mat_data = o3d.data.WoodFloorTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

	if False:
		mat_data = o3d.data.WoodTexture()

		mat = o3d.visualization.rendering.MaterialRecord()
		mat.shader = "defaultLit"
		mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
		mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
		mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)

def image_tutorial():
	img_data = o3d.data.JuneauImage()
	img = o3d.io.read_image(img_data.path)
	o3d.visualization.draw_geometries([img])

def rgbd_image_tutorial():
	if True:
		dataset = o3d.data.SampleFountainRGBDImages()

		rgbd_images = []
		for i in range(len(dataset.depth_paths)):
			depth = o3d.io.read_image(dataset.depth_paths[i])
			color = o3d.io.read_image(dataset.color_paths[i])
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
			rgbd_images.append(rgbd_image)

		camera_trajectory = o3d.io.read_pinhole_camera_trajectory(dataset.keyframe_poses_log_path)
		mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

	if False:
		import matplotlib.image as mpimg

		def read_nyu_pgm(filename, byteorder=">"):
			with open(filename, "rb") as f:
				buffer = f.read()
			try:
				header, width, height, maxval = re.search(
					b"(^P5\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n])*"
					b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer
				).groups()
			except AttributeError:
				raise ValueError("Not a raw PGM file: '%s'" % filename)
			img = np.frombuffer(
				buffer,
				dtype=byteorder + "u2",
				count=int(width) * int(height),
				offset=len(header)
			).reshape((int(height), int(width)))
			img_out = img.astype("u2")
			return img_out

		dataset = o3d.data.SampleNYURGBDImage()
		color_raw = mpimg.imread(dataset.color_path)
		depth_raw = read_nyu_pgm(dataset.depth_path)
		color = o3d.geometry.Image(color_raw)
		depth = o3d.geometry.Image(depth_raw)
		rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth, convert_rgb_to_intensity=False)
		o3d.visualization.draw_geometries([rgbd_image])

	if False:
		dataset = o3d.data.SampleRedwoodRGBDImages()

		rgbd_images = []
		for i in range(len(dataset.depth_paths)):
			color_raw = o3d.io.read_image(dataset.color_paths[i])
			depth_raw = o3d.io.read_image(dataset.depth_paths[i])
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
			rgbd_images.append(rgbd_image)

		pcd = o3d.io.read_point_cloud(dataset.reconstruction_path)
		o3d.visualization.draw(pcd)

	if False:
		dataset = o3d.data.SampleSUNRGBDImage()
		color_raw = o3d.io.read_image(dataset.color_path)
		depth_raw = o3d.io.read_image(dataset.depth_path)
		rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw, convert_rgb_to_intensity=False)
		o3d.visualization.draw_geometries([rgbd_image])

	if False:
		dataset = o3d.data.SampleTUMRGBDImage()
		color_raw = o3d.io.read_image(dataset.color_path)
		depth_raw = o3d.io.read_image(dataset.depth_path)
		rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw, convert_rgb_to_intensity=False)
		o3d.visualization.draw_geometries([rgbd_image])

	if False:
		dataset = o3d.data.LoungeRGBDImages()

		rgbd_images = []
		for i in range(len(dataset.depth_paths)):
			color_raw = o3d.io.read_image(dataset.color_paths[i])
			depth_raw = o3d.io.read_image(dataset.depth_paths[i])
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
			rgbd_images.append(rgbd_image)

		mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

	if False:
		dataset = o3d.data.BedroomRGBDImages()

		rgbd_images = []
		for i in range(len(dataset.depth_paths)):
			color_raw = o3d.io.read_image(dataset.color_paths[i])
			depth_raw = o3d.io.read_image(dataset.depth_paths[i])
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
			rgbd_images.append(rgbd_image)

		mesh = o3d.io.read_triangle_mesh(dataset.reconstruction_path)
		mesh.compute_vertex_normals()
		o3d.visualization.draw(mesh)

def demo_tutorial():
	if True:
		dataset = o3d.data.DemoColoredICPPointClouds()
		pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
		pcd1 = o3d.io.read_point_cloud(dataset.paths[1])
		o3d.visualization.draw([pcd0, pcd1])

	if False:
		dataset = o3d.data.DemoCropPointCloud()
		pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
		vol = o3d.visualization.read_selection_polygon_volume(dataset.cropped_json_path)
		chair = vol.crop_point_cloud(pcd)
		o3d.visualization.draw(chair)

	if False:
		dataset = o3d.data.DemoFeatureMatchingPointClouds()

		pcd0 = o3d.io.read_point_cloud(dataset.point_cloud_paths[0])
		pcd1 = o3d.io.read_point_cloud(dataset.point_cloud_paths[1])
		o3d.visualization.draw([pcd0, pcd1])

		fpfh_feature0 = o3d.io.read_feature(dataset.fpfh_feature_paths[0])  # open3d.pipelines.registration.Feature.
		fpfh_feature1 = o3d.io.read_feature(dataset.fpfh_feature_paths[1])  # open3d.pipelines.registration.Feature.

		l32d_feature0 = o3d.io.read_feature(dataset.l32d_feature_paths[0])  # open3d.pipelines.registration.Feature.
		l32d_feature1 = o3d.io.read_feature(dataset.l32d_feature_paths[1])  # open3d.pipelines.registration.Feature.

	if False:
		dataset = o3d.data.DemoICPPointClouds()
		pcd0 = o3d.io.read_point_cloud(dataset.paths[0])
		pcd1 = o3d.io.read_point_cloud(dataset.paths[1])
		pcd2 = o3d.io.read_point_cloud(dataset.paths[2])
		o3d.visualization.draw([pcd0, pcd1, pcd2])

	if False:
		dataset = o3d.data.DemoPoseGraphOptimization()
		pose_graph_fragment = o3d.io.read_pose_graph(dataset.pose_graph_fragment_path)  # open3d.pipelines.registration.PoseGraph.
		pose_graph_global = o3d.io.read_pose_graph(dataset.pose_graph_global_path)  # open3d.pipelines.registration.PoseGraph.

	if False:
		dataset = o3d.data.RedwoodIndoorLivingRoom1()
		assert os.path.isdir(dataset.data_root)
		pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
		o3d.visualization.draw(pcd)

		im_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_rgbds)

		im_noisy_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.noisy_depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_noisy_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_noisy_rgbds)

	if False:
		dataset = o3d.data.RedwoodIndoorLivingRoom2()
		assert os.path.isdir(dataset.data_root)
		pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
		o3d.visualization.draw(pcd)

		im_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_rgbds)

		im_noisy_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.noisy_depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_noisy_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_noisy_rgbds)

	if False:
		dataset = o3d.data.RedwoodIndoorOffice1()
		assert os.path.isdir(dataset.data_root)
		pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
		o3d.visualization.draw(pcd)

		im_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_rgbds)

		im_noisy_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.noisy_depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_noisy_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_noisy_rgbds)

	if False:
		dataset = o3d.data.RedwoodIndoorOffice2()
		assert os.path.isdir(dataset.data_root)
		pcd = o3d.io.read_point_cloud(dataset.point_cloud_path)
		o3d.visualization.draw(pcd)

		im_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_rgbds)

		im_noisy_rgbds = []
		for color_path, depth_path in zip(dataset.color_paths, dataset.noisy_depth_paths):
			im_color = o3d.io.read_image(color_path)
			im_depth = o3d.io.read_image(depth_path)
			im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth)
			im_noisy_rgbds.append(im_rgbd)
		o3d.visualization.draw_geometries(im_noisy_rgbds)

def main():
	# Dataset:
	#	REF [site] >> http://www.open3d.org/docs/latest/tutorial/data/index.html
	#	REF [site] >> https://github.com/isl-org/Open3D/tree/master/cpp/open3d/data
	#
	#	PointCloud:
	#		PCDPointCloud
	#		PLYPointCloud
	#		PTSPointCloud
	#
	#		EaglePointCloud
	#		LivingRoomPointClouds
	#		OfficePointClouds
	#
	#	TriangleMesh:
	#		ArmadilloMesh
	#		BunnyMesh
	#		KnotMesh
	#
	#	TriangleModel with PRB texture:
	#		AvocadoModel
	#		CrateModel
	#		DamagedHelmetModel
	#		FlightHelmetModel
	#		MonkeyModel
	#		SwordModel
	#
	#	Texture material images:
	#		MetalTexture
	#		PaintedPlasterTexture
	#		TerrazzoTexture
	#		TilesTexture
	#		WoodFloorTexture
	#		WoodTexture
	#
	#	Image:
	#		JuneauImage
	#
	#	RGBDImage:
	#		SampleFountainRGBDImages
	#		SampleNYURGBDImage
	#		SampleRedwoodRGBDImages
	#		SampleSUNRGBDImage
	#		SampleTUMRGBDImage
	#
	#		BedroomRGBDImages
	#		LoungeRGBDImages
	#
	#	Demo:
	#		DemoColoredICPPointClouds
	#		DemoCropPointCloud 
	#		DemoCustomVisualization
	#		DemoFeatureMatchingPointClouds
	#		DemoICPPointClouds
	#		DemoPoseGraphOptimization
	#
	#		RedwoodIndoorLivingRoom1
	#		RedwoodIndoorLivingRoom2
	#		RedwoodIndoorOffice1
	#		RedwoodIndoorOffice2
	#
	#	Others:
	#		JackJackL515Bag
	#		SampleL515Bag

	# o3d.data.Dataset:
	#	o3d.data.Dataset.data_root  # Default: $HOME/open3d_data.
	#	o3d.data.Dataset.prefix
	#	o3d.data.Dataset.download_dir  # <data_root>/download/<prefix>.
	#	o3d.data.Dataset.extract_dir  # <data_root>/extract/<prefix>.
	# o3d.data.PCDPointCloud:
	#	o3d.data.PCDPointCloud.path
	# o3d.data.SampleTUMRGBDImage:
	#	o3d.data.SampleTUMRGBDImage.color_path
	#	o3d.data.SampleTUMRGBDImage.depth_path
	# o3d.data.LoungeRGBDImages:
	#	o3d.data.LoungeRGBDImages.color_paths
	#	o3d.data.LoungeRGBDImages.depth_paths
	#	o3d.data.LoungeRGBDImages.trajectory_log_path  # Path to camera trajectory log file 'lounge_trajectory.log'.
	#	o3d.data.LoungeRGBDImages.reconstruction_path  # Path to mesh reconstruction 'lounge.ply'.
	# o3d.data.RedwoodIndoorLivingRoom1:
	#	o3d.data.RedwoodIndoorLivingRoom1.point_cloud_path  # Path to the point cloud.
	#	o3d.data.RedwoodIndoorLivingRoom1.color_paths
	#	o3d.data.RedwoodIndoorLivingRoom1.depth_paths
	#	o3d.data.RedwoodIndoorLivingRoom1.noisy_depth_paths  # Paths to the noisy depth images.
	#	o3d.data.RedwoodIndoorLivingRoom1.oni_path  # Paths to the ONI sequence.
	#	o3d.data.RedwoodIndoorLivingRoom1.trajectory_path  # Path to the ground-truth camera trajectory.
	#	o3d.data.RedwoodIndoorLivingRoom1.noise_model_path  # Path to the noise model.

	#--------------------
	#nyu_dataset_example()
	#redwood_dataset_example()
	#sun_dataset_example()
	#tum_dataset_example()

	#redwood_dataset_tutorial()
	#sun_dataset_tutorial()
	#nyu_dataset_tutorial()
	#tum_dataset_tutorial()

	#--------------------
	point_cloud_tutorial()
	triangle_mesh_tutorial()
	triangle_mesh_with_prb_texture_tutorial()
	texture_material_images_tutorial()
	image_tutorial()
	rgbd_image_tutorial()
	demo_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
