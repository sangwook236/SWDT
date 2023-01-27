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

def main():
	# REF [site] >> https://github.com/isl-org/Open3D/tree/master/cpp/open3d/data
	#	AvocadoModel
	#	BedroomRGBDImages
	#	BunnyMesh
	#	CrateModel
	#	DamagedHelmetModel
	#	EaglePointCloud
	#	FlightHelmetModel
	#	JackJackL515Bag
	#	JuneauImage
	#	KnotMesh
	#	LivingRoomPointClouds
	#	LoungeRGBDImages
	#	MonkeyModel
	#	OfficePointClouds
	#	PCDPointCloud
	#	PLYPointCloud
	#	PTSPointCloud
	#	RedwoodIndoorLivingRoom1
	#	RedwoodIndoorLivingRoom2
	#	RedwoodIndoorOffice1
	#	RedwoodIndoorOffice2
	#
	#	DemoColoredICPPointClouds 
	#	DemoCropPointCloud 
	#	DemoCustomVisualization
	#	DemoFeatureMatchingPointClouds
	#	DemoICPPointClouds
	#	DemoPoseGraphOptimization
	#
	#	SampleFountainRGBDImages
	#	SampleL515Bag
	#	SampleNYURGBDImage
	#	SampleRedwoodRGBDImages
	#	SampleSUNRGBDImage
	#	SampleTUMRGBDImage
	#	SwordModel
	#
	#	MetalTexture
	#	PaintedPlasterTexture
	#	TerrazzoTexture
	#	TilesTexture
	#	WoodFloorTexture
	#	WoodTexture

	#nyu_dataset_example()
	#redwood_dataset_example()
	#sun_dataset_example()
	tum_dataset_example()

	#redwood_dataset_tutorial()
	#sun_dataset_tutorial()
	#nyu_dataset_tutorial()
	#tum_dataset_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
