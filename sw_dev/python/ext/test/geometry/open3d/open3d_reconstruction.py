#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import open3d as o3d
#import open3d_tutorial as o3dtut
import matplotlib.pyplot as plt
import reconstruction_system as recon_sys

# REF [site] >>
#	http://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html
#	https://github.com/isl-org/Open3D/tree/master/examples/python/reconstruction_system
def reconstruction_system_tutuiral():
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

def main():
	reconstruction_system_tutuiral()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
