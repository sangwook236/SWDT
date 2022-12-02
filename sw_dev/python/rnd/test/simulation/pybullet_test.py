#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import pybullet as p
import pybullet_data

# REF [doc] >> "PyBullet Quickstart Guide".
def hello_pubullet_world_introduction():
	physicsClient = p.connect(p.GUI)  # Or p.DIRECT for non-graphical version.

	p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Optionally.
	p.setGravity(0, 0, -10)

	# REF [site] >> https://github.com/bulletphysics/bullet3/tree/master/data
	planeId = p.loadURDF("./plane.urdf")
	startPos = [0, 0, 1]
	startOrientation = p.getQuaternionFromEuler([0, 0, 0])
	boxId = p.loadURDF("./r2d2.urdf", startPos, startOrientation)

	# Set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation).
	for i in range(10000):
		p.stepSimulation()
		time.sleep(1.0 / 240.0)

	cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
	print(cubePos, cubeOrn)

	p.disconnect()

def main():
	hello_pubullet_world_introduction()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
