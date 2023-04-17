#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import open3d as o3d

def fpfh_test():
	pcd = o3d.io.read_point_cloud("/path/to/pcloud.ply")
	print(f"#points = {len(pcd.points)}.")

	# Downsample.
	voxel_size = 10
	pcd_down = pcd.voxel_down_sample(voxel_size)
	print(f"#points downsampled = {len(pcd_down.points)}.")

	# Extract features.
	radius_feature = voxel_size * 5
	pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
		pcd_down,
		o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
	)

	print(f"Dimension = {pcd_fpfh.dimension()}.")  # 33.
	print(f"#feature points = {pcd_fpfh.num()}.")  # #feature points = #input points.
	print(f"Data: shape = {pcd_fpfh.data.shape}, dtype = {pcd_fpfh.data.dtype}.")  # (feature dimension, #feature points).

def main():
	fpfh_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
