#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
import numpy as np
import open3d as o3d
#import open3d_tutorial as o3dtut
import matplotlib.pyplot as plt

# REF [site] >> http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
def point_cloud_tutorial():
	print("Load a ply point cloud, print it, and render it.")
	pcd = o3d.io.read_point_cloud("./test_data/fragment.ply")
	#print(pcd)
	#print(np.asarray(pcd.points))
	print("#points = {}.".format(len(pcd.points)))
	print("#points = {}.".format(np.asarray(pcd.points).shape))

	# Visualize point cloud.
	# It looks like a dense surface, but it is actually a point cloud rendered as surfels.
	o3d.visualization.draw_geometries([pcd],
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

	"""
	mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
	mesh = mesh[0]
	print("Try to render a mesh with normals (exist: " +
		str(mesh.has_vertex_normals()) + ") and colors (exist: " +
		str(mesh.has_vertex_colors()) + ")")
	o3d.visualization.draw_geometries([mesh])
	print("A mesh with no normals and no colors does not look good.")
	"""

	# Voxel downsampling.
	print("Downsample the point cloud with a voxel of 0.05.")
	downpcd = pcd.voxel_down_sample(voxel_size=0.05)
	o3d.visualization.draw_geometries([downpcd],
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

	# Vertex normal estimation.
	print("Recompute the normal of the downsampled point cloud.")
	downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	o3d.visualization.draw_geometries([downpcd],
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024],
		point_show_normal=True
	)

	# Access estimated vertex normal.
	print("Print a normal vector of the 0th point.")
	print(downpcd.normals[0])

	print("Print the normal vectors of the first 10 points.")
	print(np.asarray(downpcd.normals)[:10, :])

	# Crop point cloud.
	print("Load a polygon volume and use it to crop the original point cloud.")
	vol = o3d.visualization.read_selection_polygon_volume("./test_data/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)
	o3d.visualization.draw_geometries([chair],
		zoom=0.7,
		front=[0.5439, -0.2333, -0.8060],
		lookat=[2.4615, 2.1331, 1.338],
		up=[-0.1781, -0.9708, 0.1608]
	)

	# Paint point cloud.
	print("Paint chair.")
	chair.paint_uniform_color([1, 0.706, 0])
	o3d.visualization.draw_geometries([chair],
		zoom=0.7,
		front=[0.5439, -0.2333, -0.8060],
		lookat=[2.4615, 2.1331, 1.338],
		up=[-0.1781, -0.9708, 0.1608]
	)

	# Point cloud distance.
	# Load data
	pcd = o3d.io.read_point_cloud("./test_data/fragment.ply")
	vol = o3d.visualization.read_selection_polygon_volume("./test_data/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)

	dists = pcd.compute_point_cloud_distance(chair)
	dists = np.asarray(dists)
	ind = np.where(dists > 0.01)[0]
	pcd_without_chair = pcd.select_by_index(ind)
	o3d.visualization.draw_geometries([pcd_without_chair],
		zoom=0.3412,
		front=[0.4257, -0.2125, -0.8795],
		lookat=[2.6172, 2.0475, 1.532],
		up=[-0.0694, -0.9768, 0.2024]
	)

	# Bounding volumes.
	aabb = chair.get_axis_aligned_bounding_box()
	aabb.color = (1, 0, 0)
	obb = chair.get_oriented_bounding_box()
	obb.color = (0, 1, 0)
	o3d.visualization.draw_geometries([chair, aabb, obb],
		zoom=0.7,
		front=[0.5439, -0.2333, -0.8060],
		lookat=[2.4615, 2.1331, 1.338],
		up=[-0.1781, -0.9708, 0.1608]
	)

	# Convex hull.
	if False:
		pcl = o3dtut.get_bunny_mesh().sample_points_poisson_disk(number_of_points=2000)
	else:
		# REF [site] >> http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz
		mesh = o3d.io.read_triangle_mesh("./test_data/bun_zipper.ply")
		mesh.compute_vertex_normals()
		pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
	hull, _ = pcl.compute_convex_hull()
	hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
	hull_ls.paint_uniform_color((1, 0, 0))
	o3d.visualization.draw_geometries([pcl, hull_ls])

	# DBSCAN clustering.
	pcd = o3d.io.read_point_cloud("./test_data/fragment.ply")

	with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
		labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

	max_label = labels.max()
	print(f"point cloud has {max_label + 1} clusters")
	colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
	colors[labels < 0] = 0
	pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
	o3d.visualization.draw_geometries([pcd],
		zoom=0.455,
		front=[-0.4999, -0.1659, -0.8499],
		lookat=[2.1813, 2.0619, 2.0999],
		up=[0.1204, -0.9852, 0.1215]
	)

	# Plane segmentation.
	pcd = o3d.io.read_point_cloud("./test_data/fragment.pcd")
	plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
	[a, b, c, d] = plane_model
	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

	inlier_cloud = pcd.select_by_index(inliers)
	inlier_cloud.paint_uniform_color([1.0, 0, 0])
	outlier_cloud = pcd.select_by_index(inliers, invert=True)
	o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
		zoom=0.8,
		front=[-0.4999, -0.1659, -0.8499],
		lookat=[2.1813, 2.0619, 2.0999],
		up=[0.1204, -0.9852, 0.1215]
	)

	# Hidden point removal.
	print("Convert mesh to a point cloud and estimate dimensions.")
	if False:
		pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(5000)
	else:
		# REF [site] >> http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz
		mesh = o3d.io.read_triangle_mesh("./test_data/Armadillo.ply")
		mesh.compute_vertex_normals()
		pcd = mesh.sample_points_poisson_disk(5000)
	diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
	o3d.visualization.draw_geometries([pcd])

	print("Define parameters used for hidden_point_removal.")
	camera = [0, 0, diameter]
	radius = diameter * 100

	print("Get all points that are visible from given view point.")
	_, pt_map = pcd.hidden_point_removal(camera, radius)

	print("Visualize result.")
	pcd = pcd.select_by_index(pt_map)
	o3d.visualization.draw_geometries([pcd])

# REF [site] >> http://www.open3d.org/docs/release/tutorial/geometry/mesh.html
def mesh_tutorial():
	print("Testing mesh in Open3D...")
	if False:
		mesh = o3dtut.get_knot_mesh()
	else:
		mesh = o3d.io.read_triangle_mesh("./test_data/knot.ply")
		mesh.compute_vertex_normals()
	#print(mesh)
	print("#vertices = {}.".format(np.asarray(mesh.vertices).shape))
	print("#triangles = {}.".format(np.asarray(mesh.triangles).shape))

	# Visualize a 3D mesh.
	print("Try to render a mesh with normals (exist: " +
		str(mesh.has_vertex_normals()) + ") and colors (exist: " +
		str(mesh.has_vertex_colors()) + ")")
	o3d.visualization.draw_geometries([mesh])
	print("A mesh with no normals and no colors does not look good.")

	# Surface normal estimation.
	print("Computing normal and rendering it.")
	mesh.compute_vertex_normals()
	print(np.asarray(mesh.triangle_normals))
	o3d.visualization.draw_geometries([mesh])

	# Crop mesh.
	print("We make a partial mesh of only the first half triangles.")
	mesh1 = copy.deepcopy(mesh)
	mesh1.triangles = o3d.utility.Vector3iVector(np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
	mesh1.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
	print(mesh1.triangles)
	o3d.visualization.draw_geometries([mesh1])

	# Paint mesh.
	print("Painting the mesh.")
	mesh1.paint_uniform_color([1, 0.706, 0])
	o3d.visualization.draw_geometries([mesh1])

def convert_file_format():
	input_filepath = "/path/to/input.pcd"
	output_filepath = "/path/to/output.ply"

	pc = o3d.io.read_point_cloud(input_filepath)
	o3d.io.write_point_cloud(output_filepath, pc)

def main():
	#point_cloud_tutorial()
	#mesh_tutorial()

	convert_file_format()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
