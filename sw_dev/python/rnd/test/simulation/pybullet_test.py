#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, random, glob, time
import numpy as np
import pybullet as p
import pybullet_data

# REF [doc] >> "PyBullet Quickstart Guide".
def hello_pybullet_world_introduction():
	#physicsClient = p.connect(p.GUI)
	#physicsClient = p.connect(p.GUI, options="--opengl2")
	physicsClient = p.connect(p.DIRECT)  # For non-graphical version.

	if True:
		print(f"p.isConnected() = {p.isConnected()}.")
		print(f"p.getConnectionInfo() = {p.getConnectionInfo()}.")

		print(f"p.getPhysicsEngineParameters() = {p.getPhysicsEngineParameters()}.")
		print(f"p.isNumpyEnabled() = {p.isNumpyEnabled()}.")

	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	#p.configureDebugVisualizer(flag=p.ENABLE_RENDERING, enable=False)
	#p.configureDebugVisualizer(flag=p.COV_ENABLE_SINGLE_STEP_RENDERING, enable=True)

	#p.setRealTimeSimulation(enableRealTimeSimulation=True)
	#p.resetSimulation(flags=p.RESET_USE_DEFORMABLE_WORLD)
	#p.setTimeOut(4000000)  # [sec].
	#p.setTimeStep(1 / 240)  # [Hz].
	p.setGravity(0, 0, -10)

	#loggingId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName="/path/to/log")
	#p.stopStateLogging(loggingUniqueId=loggingId)

	#logId = p.startStateLogging(loggingType=p.STATE_LOGGING_PROFILE_TIMINGS, fileName="/path/to/file.json")
	#p.submitProfileTiming("profiling_name")
	## Do something.
	#p.submitProfileTiming()

	# REF [site] >> https://github.com/bulletphysics/bullet3/tree/master/data
	planeId = p.loadURDF("plane.urdf")
	startPos = [0, 0, 1]
	startOrientation = p.getQuaternionFromEuler([0, 0, 0])
	boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

	if True:
		print("------------------------------------------------------------")
		numBodies = p.getNumBodies()
		print(f"p.getNumBodies() = {numBodies}.")
		for bodyIndex in range(numBodies):
			print("--------------------")
			print(f"p.getBodyInfo(bodyUniqueId={bodyIndex}) = {p.getBodyInfo(bodyUniqueId=bodyIndex)}.")
			pos, orient = p.getBasePositionAndOrientation(bodyUniqueId=bodyIndex)
			orient_euler = p.getEulerFromQuaternion(orient)
			print(f"p.getBasePositionAndOrientation(bodyUniqueId={bodyIndex}) = {(pos, orient)}, p.getEulerFromQuaternion({orient}) = {orient_euler}.")
			print(f"p.getBaseVelocity(bodyUniqueId={bodyIndex}) = {p.getBaseVelocity(bodyUniqueId=bodyIndex)}.")

			numJoints = p.getNumJoints(bodyUniqueId=bodyIndex)
			print(f"p.getNumJoints(bodyUniqueId={bodyIndex}) = {numJoints}.")
			for jointIndex in range(numJoints):
				print("----------")
				print(f"p.getJointInfo(bodyUniqueId={bodyIndex}, jointIndex={jointIndex}) = {p.getJointInfo(bodyUniqueId=bodyIndex, jointIndex=jointIndex)}.")
				print(f"p.getJointState(bodyUniqueId={bodyIndex}, jointIndex={jointIndex}) = {p.getJointState(bodyUniqueId=bodyIndex, jointIndex=jointIndex)}.")

				print(f"p.getLinkState(bodyUniqueId={bodyIndex}, linkIndex={jointIndex}, computeLinkVelocity=True, computeForwardKinematics=True) = {p.getLinkState(bodyUniqueId=bodyIndex, linkIndex=jointIndex, computeLinkVelocity=True, computeForwardKinematics=True)}.")

				print(f"p.getDynamicsInfo(bodyUniqueId={bodyIndex}, linkIndex={jointIndex}) = {p.getDynamicsInfo(bodyUniqueId=bodyIndex, linkIndex=jointIndex)}.")

		print("------------------------------")
		print(f"p.getNumConstraints() = {p.getNumConstraints()}.")
		for constraintIndex in range(p.getNumConstraints()):
			print("--------------------")
			print(f"p.getConstraintInfo(constraintUniqueId={constraintIndex}) = {p.getConstraintInfo(constraintUniqueId=constraintIndex)}.")
			print(f"p.getConstraintState(constraintUniqueId={constraintIndex}) = {p.getConstraintState(constraintUniqueId=constraintIndex)}.")

	print("------------------------------")
	# Set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation).
	for i in range(10000):
		p.stepSimulation()
		#time.sleep(1.0 / 240.0)

	print(f"p.getBasePositionAndOrientation(boxId) = {p.getBasePositionAndOrientation(boxId)}.")

	p.disconnect()

# REF [site] >> https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
def simulate_camera_images_test():
	physicsClient = p.connect(p.GUI)

	#-----
	# Load a plane.
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	planeId = p.loadURDF("plane.urdf")

	# Load objects.
	# Visual shape.
	visualShapeId = p.createVisualShape(
		shapeType=p.GEOM_MESH,
		fileName="random_urdfs/000/000.obj",
		rgbaColor=None,
		meshScale=[0.1, 0.1, 0.1],
	)
	assert visualShapeId >= 0

	# Collision shape.
	collisionShapeId = p.createCollisionShape(
		shapeType=p.GEOM_MESH,
		#fileName="random_urdfs/000/000_coll.obj",  # No file exists.
		fileName="random_urdfs/000/000.obj",
		meshScale=[0.1, 0.1, 0.1],
	)
	assert collisionShapeId >= 0

	# Multi-body.
	multiBodyId = p.createMultiBody(
		baseMass=1.0,
		baseCollisionShapeIndex=collisionShapeId, 
		baseVisualShapeIndex=visualShapeId,
		basePosition=[0, 0, 1],
		baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
	)
	assert multiBodyId >= 0

	#-----
	# Apply textures to objects.
	#	REF [site] >> https://www.robots.ox.ac.uk/~vgg/data/dtd/
	texture_paths = glob.glob(os.path.join("dtd", "**", "*.jpg"), recursive=True)
	random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
	textureId = p.loadTexture(random_texture_path)
	assert textureId >= 0

	p.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)

	#-----
	# Simulation.
	p.setGravity(0, 0, -9.8)
	p.setRealTimeSimulation(1)

	#-----
	# Render images.

	# View matrix.
	viewMatrix = p.computeViewMatrix(
		cameraEyePosition=[0, 0, 3],
		cameraTargetPosition=[0, 0, 0],
		cameraUpVector=[0, 1, 0],
	)

	# Projection matrix.
	projectionMatrix = p.computeProjectionMatrixFOV(
		fov=45.0,
		aspect=1.0,
		nearVal=0.1,
		farVal=3.1,
	)

	# Get camera images.
	imageWidth, imageHeight, rgbaPixels, depthPixels, segmentationMaskBuffer = p.getCameraImage(
		width=224, 
		height=224,
		viewMatrix=viewMatrix,
		projectionMatrix=projectionMatrix,
	)

	rgbaImage = np.reshape(rgbaPixels, (imageHeight, imageWidth, 4))  # RGBA.
	depthImage = np.reshape(depthPixels, (imageHeight, imageWidth))
	segMaskImage = np.reshape(segmentationMaskBuffer, (imageHeight, imageWidth))

	print(f"RGBA image: shape = {rgbaImage.shape}, dtype = {rgbaImage.dtype}, (min, max) = ({np.min(rgbaImage)}, {np.max(rgbaImage)}).")
	print(f"Depth image: shape = {depthImage.shape}, dtype = {depthImage.dtype}, (min, max) = ({np.min(depthImage)}, {np.max(depthImage)}).")
	print(f"Segmentation mask: shape = {segMaskImage.shape}, dtype = {segMaskImage.dtype}, (min, max) = ({np.min(segMaskImage)}, {np.max(segMaskImage)}).")

	if False:
		import cv2

		cv2.imwrite("./rgba.png", rgbaImage)
		cv2.imwrite("./depth.png", (depthImage * 255).astype(np.uint8))
		cv2.imwrite("./seg_mask.png", segMaskImage * 255)

	# REF [site] >> https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py

	#-----
	# Run simulation.
	print("Start simulation...")
	while True:
		p.stepSimulation()
	print("End simulation.")

# REF [site] >> https://github.com/ElectronicElephant/pybullet_ur5_robotiq
def articulated_robot_test():
	from collections import namedtuple

	model_dir_path = './models'

	physicsClient = p.connect(p.GUI)

	#p.setGravity(0, 0, -9.8)
	#p.setRealTimeSimulation(1)

	#-----
	# Load a plane.
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	planeId = p.loadURDF("plane.urdf")

	#-----
	# Load a robot.
	# REF [function] >> RobotBase.__parse_joint_info__() in https://github.com/ElectronicElephant/pybullet_ur5_robotiq/blob/robotflow/robot.py

	def parse_joint_info(robotId, robotDof):
		numJoints = p.getNumJoints(robotId)
		jointInfo = namedtuple("jointInfo", ["id", "name", "type", "damping", "friction", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
		joints = []
		controllable_joints = []
		for i in range(numJoints):
			info = p.getJointInfo(robotId, i)
			jointID = info[0]
			jointName = info[1].decode("utf-8")
			jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
			jointDamping = info[6]
			jointFriction = info[7]
			jointLowerLimit = info[8]
			jointUpperLimit = info[9]
			jointMaxForce = info[10]
			jointMaxVelocity = info[11]
			controllable = (jointType != p.JOINT_FIXED)
			if controllable:
				controllable_joints.append(jointID)
				p.setJointMotorControl2(robotId, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
			info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit, jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
			joints.append(info)

		assert len(controllable_joints) >= robotDof
		arm_controllable_joints = controllable_joints[:robotDof]

		arm_lower_limits = [info.lowerLimit for info in joints if info.controllable][:robotDof]
		arm_upper_limits = [info.upperLimit for info in joints if info.controllable][:robotDof]
		arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in joints if info.controllable][:robotDof]

		return joints, controllable_joints, arm_controllable_joints, arm_lower_limits, arm_upper_limits, arm_joint_ranges

	robot_base_position = (0, 0.5, 0)
	robot_base_orientation = p.getQuaternionFromEuler((0, 0, 0))  # (roll, pitch, yaw).
	if True:
		# UR5 + ROBOTIQ 85.
		robotId = p.loadURDF(os.path.join(model_dir_path, "urdf/ur5_robotiq_85.urdf"), robot_base_position, robot_base_orientation, useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
		endEffectorId = 7
		robotDof = 6
		armRestPoses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636]
	elif False:
		# UR5 + ROBOTIQ 140.
		robotId = p.loadURDF(os.path.join(model_dir_path, "urdf/ur5_robotiq_140.urdf"), robot_base_position, robot_base_orientation, useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
		endEffectorId = 7
		robotDof = 6
		armRestPoses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636]
	else:
		# Panda (Franka Emika).
		robotId = p.loadURDF(os.path.join(model_dir_path, "urdf/panda.urdf"), robot_base_position, robot_base_orientation, useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
		endEffectorId = 11
		robotDof = 7
		armRestPoses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]

	#joints, controllable_joints, arm_controllable_joints, arm_lower_limits, arm_upper_limits, arm_joint_ranges = parse_joint_info(robotId, robotDof)
	arm_controllable_joints = parse_joint_info(robotId, robotDof)[2]

	for restPose, jointId in zip(armRestPoses, arm_controllable_joints):
		p.resetJointState(robotId, jointId, restPose)

	#-----
	# Load a box.
	boxID = p.loadURDF(
		os.path.join(model_dir_path, "urdf/skew-box-button.urdf"),
		(0.0, 0.0, 0.0),
		#p.getQuaternionFromEuler((0, 1.5706453, 0)),
		p.getQuaternionFromEuler((0, 0, 0)),
		useFixedBase=True,
		flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION
	)

	p.setJointMotorControl2(boxID, 0, p.POSITION_CONTROL, force=1)
	p.setJointMotorControl2(boxID, 1, p.VELOCITY_CONTROL, force=0)

	#-----
	# Render images.

	# View matrix.
	viewMatrix = p.computeViewMatrix(
		cameraEyePosition=(1, 1, 1),
		cameraTargetPosition=(0, 0, 0),
		cameraUpVector=(0, 0, 1),
	)

	# Projection matrix.
	projectionMatrix = p.computeProjectionMatrixFOV(
		fov=40.0,
		aspect=1.0,
		nearVal=0.1,
		farVal=5.0,
	)

	# Get camera images.
	imageWidth, imageHeight, rgbaPixels, depthPixels, segmentationMaskBuffer = p.getCameraImage(
		width=320, 
		height=320,
		viewMatrix=viewMatrix,
		projectionMatrix=projectionMatrix,
	)

	rgbaImage = np.reshape(rgbaPixels, (imageHeight, imageWidth, 4))  # RGBA.
	depthImage = np.reshape(depthPixels, (imageHeight, imageWidth))
	segMaskImage = np.reshape(segmentationMaskBuffer, (imageHeight, imageWidth))

	print(f"RGBA image: shape = {rgbaImage.shape}, dtype = {rgbaImage.dtype}, (min, max) = ({np.min(rgbaImage)}, {np.max(rgbaImage)}).")
	print(f"Depth image: shape = {depthImage.shape}, dtype = {depthImage.dtype}, (min, max) = ({np.min(depthImage)}, {np.max(depthImage)}).")
	print(f"Segmentation mask: shape = {segMaskImage.shape}, dtype = {segMaskImage.dtype}, (min, max) = ({np.min(segMaskImage)}, {np.max(segMaskImage)}).")

	#-----
	# Run simulation.
	print("Start simulation...")
	while True:
		p.stepSimulation()
	print("End simulation.")

# REF [site] >>
#	https://www.ycbbenchmarks.com/
#	http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
def ycb_benchmark_test():
	model_dir_path = './models'

	physicsClient = p.connect(p.GUI)

	#p.setGravity(0, 0, -9.8)
	#p.setRealTimeSimulation(1)

	#-----
	# Load a plane.
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	planeId = p.loadURDF("plane.urdf")

	#-----
	# Load an object.

	# REF [site] >> https://github.com/harvard-microrobotics/object2urdf
	object_urdf = os.path.join(model_dir_path, "ycb/006_mustard_bottle.urdf")
	#object_urdf = os.path.join(model_dir_path, "ycb/013_apple.urdf")
	#object_urdf = os.path.join(model_dir_path, "ycb/025_mug.urdf")
	#object_urdf = os.path.join(model_dir_path, "ycb/053_mini_soccer_ball.urdf")
	objectId = p.loadURDF(
		fileName=object_urdf,
		basePosition=[0, 0, 2],
		baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
		useFixedBase=True,
	)

	#-----
	# Render images.

	# View matrix.
	viewMatrix = p.computeViewMatrix(
		cameraEyePosition=(2, 2, 3),
		cameraTargetPosition=(0, 0, 1),
		cameraUpVector=(0, 0, 1),
	)

	# Projection matrix.
	projectionMatrix = p.computeProjectionMatrixFOV(
		fov=40.0,
		aspect=1.0,
		nearVal=0.1,
		farVal=5.0,
	)

	# Get camera images.
	imageWidth, imageHeight, rgbaPixels, depthPixels, segmentationMaskBuffer = p.getCameraImage(
		width=320, 
		height=320,
		viewMatrix=viewMatrix,
		projectionMatrix=projectionMatrix,
	)

	rgbaImage = np.reshape(rgbaPixels, (imageHeight, imageWidth, 4))  # RGBA.
	depthImage = np.reshape(depthPixels, (imageHeight, imageWidth))
	segMaskImage = np.reshape(segmentationMaskBuffer, (imageHeight, imageWidth))

	print(f"RGBA image: shape = {rgbaImage.shape}, dtype = {rgbaImage.dtype}, (min, max) = ({np.min(rgbaImage)}, {np.max(rgbaImage)}).")
	print(f"Depth image: shape = {depthImage.shape}, dtype = {depthImage.dtype}, (min, max) = ({np.min(depthImage)}, {np.max(depthImage)}).")
	print(f"Segmentation mask: shape = {segMaskImage.shape}, dtype = {segMaskImage.dtype}, (min, max) = ({np.min(segMaskImage)}, {np.max(segMaskImage)}).")

	#-----
	# Run simulation.
	print("Start simulation...")
	while True:
		p.stepSimulation()
	print("End simulation.")

def environment_test():
	import pybullet_envs  # Register PyBullet environments.

	print(f"Environments: {pybullet_envs.getList()}.")

	print("----------")
	if False:
		#import gym
		#env = gym.make("MinitaurBulletEnv-v0")

		import pybullet_envs.bullet.minitaur_gym_env as e
		env = e.MinitaurBulletEnv(action_repeat=1, render=False)
	elif True:
		import pybullet_envs.bullet.racecarGymEnv as e
		env = e.RacecarGymEnv(actionRepeat=50, isEnableSelfCollision=True, isDiscrete=False, renders=False)
		env.reset()
	elif False:
		import pybullet_envs.bullet.kukaGymEnv as e
		env = e.KukaGymEnv(actionRepeat=1, isEnableSelfCollision=True, isDiscrete=False, renders=False, maxSteps=1000)
		env.reset()

	print("Start simulation...")
	for _ in range(10000):
		p.stepSimulation()
	print("End simulation.")

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/bullet/racecarGymEnv.py
def racecar_gym_env_example():
	from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

	isDiscrete = False

	environment = RacecarGymEnv(renders=True, isDiscrete=isDiscrete)
	environment.reset()

	targetVelocitySlider = environment._p.addUserDebugParameter("wheelVelocity", -1, 1, 0)
	steeringSlider = environment._p.addUserDebugParameter("steering", -1, 1, 0)

	while (True):
		targetVelocity = environment._p.readUserDebugParameter(targetVelocitySlider)
		steeringAngle = environment._p.readUserDebugParameter(steeringSlider)
		if (isDiscrete):
			discreteAction = 0
			if (targetVelocity < -0.33):
				discreteAction = 0
			else:
				if (targetVelocity > 0.33):
					discreteAction = 6
				else:
					discreteAction = 3
			if (steeringAngle > -0.17):
				if (steeringAngle > 0.17):
					discreteAction = discreteAction + 2
				else:
					discreteAction = discreteAction + 1
			action = discreteAction
		else:
			action = [targetVelocity, steeringAngle]
		state, reward, done, info = environment.step(action)
		obs = environment.getExtendedObservation()
		print("obs")
		print(obs)

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/bullet/kukaGymEnv.py
def kuka_gym_env_example():
	from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

	environment = KukaGymEnv(renders=True, isDiscrete=False, maxSteps=10000000)

	motorsIds = []
	#motorsIds.append(environment._p.addUserDebugParameter("posX", 0.4, 0.75, 0.537))
	#motorsIds.append(environment._p.addUserDebugParameter("posY", -0.22, 0.3, 0.0))
	#motorsIds.append(environment._p.addUserDebugParameter("posZ", 0.1, 1, 0.2))
	#motorsIds.append(environment._p.addUserDebugParameter("yaw", -3.14, 3.14, 0))
	#motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, 0.3))
	dv = 0.01
	motorsIds.append(environment._p.addUserDebugParameter("posX", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("posY", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("posZ", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("yaw", -dv, dv, 0))
	motorsIds.append(environment._p.addUserDebugParameter("fingerAngle", 0, 0.3, 0.3))

	done = False
	while (not done):
		action = []
		for motorId in motorsIds:
			action.append(environment._p.readUserDebugParameter(motorId))

		state, reward, done, info = environment.step2(action)
		obs = environment.getExtendedObservation()

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/train_kuka_grasping.py
def baselines_train_kuka_grasping_example():
	from baselines import deepq
	from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

	def callback(lcl, glb):
		# Stop training if reward exceeds 199.
		total = sum(lcl["episode_rewards"][-101:-1]) / 100
		totalt = lcl["t"]
		#print("totalt")
		#print(totalt)
		is_solved = totalt > 2000 and total >= 10
		return is_solved

	env = KukaGymEnv(renders=False, isDiscrete=True)
	model = deepq.models.mlp([64])
	act = deepq.learn(
		env,
		q_func=model,
		lr=1e-3,
		max_timesteps=10000000,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.02,
		print_freq=10,
		callback=callback,
	)
	print("Saving model to kuka_model.pkl.")
	act.save("kuka_model.pkl")

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_kuka_grasping.py
def baselines_enjoy_kuka_grasping_example():
	from baselines import deepq
	from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

	env = KukaGymEnv(renders=True, isDiscrete=True)
	act = deepq.load("kuka_model.pkl")
	print(act)
	while True:
		obs, done = env.reset(), False
		print("===================================")
		print("obs")
		print(obs)
		episode_rew = 0
		while not done:
			env.render()
			obs, rew, done, _ = env.step(act(obs[None])[0])
			episode_rew += rew
		print("Episode reward", episode_rew)

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/train_pybullet_cartpole.py
def baselines_train_pybullet_cartpole_example():
	from baselines import deepq
	from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv

	def callback(lcl, glb):
		# Stop training if reward exceeds 199.
		is_solved = lcl["t"] > 100 and sum(lcl["episode_rewards"][-101:-1]) / 100 >= 199
		return is_solved

	env = CartPoleBulletEnv(renders=False)
	model = deepq.models.mlp([64])
	act = deepq.learn(
		env,
		q_func=model,
		lr=1e-3,
		max_timesteps=100000,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.02,
		print_freq=10,
		callback=callback,
	)
	print("Saving model to cartpole_model.pkl.")
	act.save("cartpole_model.pkl")

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_pybullet_cartpole.py
def baselines_enjoy_pybullet_cartpole_example():
	#import gym
	from baselines import deepq
	from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv

	#env = gym.make("CartPoleBulletEnv-v1")
	env = CartPoleBulletEnv(renders=True, discrete_actions=True)
	act = deepq.load("cartpole_model.pkl")

	while True:
		obs, done = env.reset(), False
		print("obs")
		print(obs)
		print("type(obs)")
		print(type(obs))
		episode_rew = 0
		while not done:
			env.render()

			o = obs[None]
			aa = act(o)
			a = aa[0]
			obs, rew, done, _ = env.step(a)
			episode_rew += rew
			time.sleep(1. / 240.)
		print("Episode reward", episode_rew)

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/train_pybullet_racecar.py
def baselines_train_pybullet_racecar_example():
	from baselines import deepq
	from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

	def callback(lcl, glb):
		# Stop training if reward exceeds 199.
		total = sum(lcl["episode_rewards"][-101:-1]) / 100
		totalt = lcl["t"]
		is_solved = totalt > 2000 and total >= -50
		return is_solved

	env = RacecarGymEnv(renders=False, isDiscrete=True)
	model = deepq.models.mlp([64])
	act = deepq.learn(
		env,
		q_func=model,
		lr=1e-3,
		max_timesteps=10000,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.02,
		print_freq=10,
		callback=callback,
	)
	print("Saving model to racecar_model.pkl.")
	act.save("racecar_model.pkl")

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines/enjoy_pybullet_racecar.py
def baselines_enjoy_pybullet_racecar_example():
	from baselines import deepq
	from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

	env = RacecarGymEnv(renders=True, isDiscrete=True)
	act = deepq.load("racecar_model.pkl")
	print(act)
	while True:
		obs, done = env.reset(), False
		print("===================================")
		print("obs")
		print(obs)
		episode_rew = 0
		while not done:
			env.render()
			obs, rew, done, _ = env.step(act(obs[None])[0])
			episode_rew += rew
		print("Episode reward", episode_rew)

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/stable_baselines/train.py
def stable_baselines_train_example():
	import numpy as np
	import gym
	from stable_baselines3 import SAC, TD3
	from stable_baselines3.common.noise import NormalActionNoise
	from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
	from stable_baselines3.common.monitor import Monitor
	import pybullet_envs  # Register PyBullet envs.

	if True:
		algo_id = "sac"
	elif False:
		algo_id = "td3"
	env_id = "HalfCheetahBulletEnv-v0"  # Environment ID.
	n_timesteps = int(1e6)  # Number of training timesteps.
	save_freq = -1  # Save the model every n steps (if negative, no checkpoint).
	save_path = f"{algo_id}_{env_id}"

	# Instantiate and wrap the environment.
	env = gym.make(env_id)

	# Create the evaluation environment and callbacks.
	eval_env = Monitor(gym.make(env_id))

	callbacks = [EvalCallback(eval_env, best_model_save_path=save_path)]

	# Save a checkpoint every n steps.
	if save_freq > 0:
		callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=save_path, name_prefix="rl_model"))

	n_actions = env.action_space.shape[0]

	# Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
	if algo_id == "sac":
		algo = SAC  # RL Algorithm.
		hyperparams = dict(
			batch_size=256,
			gamma=0.98,
			policy_kwargs=dict(net_arch=[256, 256]),
			learning_starts=10000,
			buffer_size=int(3e5),
			tau=0.01,
		)
	elif algo_id == "td3":
		algo = TD3  # RL Algorithm.
		hyperparams = dict(
			batch_size=100,
			policy_kwargs=dict(net_arch=[400, 300]),
			learning_rate=1e-3,
			learning_starts=10000,
			buffer_size=int(1e6),
			train_freq=1,
			gradient_steps=1,
			action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
		)

	model = algo("MlpPolicy", env, verbose=1, **hyperparams)
	try:
		model.learn(n_timesteps, callback=callbacks)
	except KeyboardInterrupt:
		pass

	print(f"Saving to {save_path}.zip.")
	model.save(save_path)

# REF [file] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/enjoy.py
def stable_baselines_enjoy_example():
	import numpy as np
	import gym
	from stable_baselines3 import SAC, TD3
	import pybullet_envs  # Register PyBullet envs.

	if True:
		algo = SAC  # RL Algorithm.
		algo_id = "sac"
	elif False:
		algo = TD3  # RL Algorithm.
		algo_id = "td3"
	env_id = "HalfCheetahBulletEnv-v0"  # Environment ID.
	n_episodes = 5  # Number of episodes.
	no_render = False  # Do not render the environment.
	load_best = False  # Load best model instead of last model if available.

	# Create an env similar to the training env.
	env = gym.make(env_id)

	# Enable GUI.
	if not no_render:
		env.render(mode="human")

	# We assume that the saved model is in the same folder.
	save_path = f"{algo_id}_{env_id}.zip"

	if not os.path.isfile(save_path) or load_best:
		print("Loading best model.")
		# Try to load best model.
		save_path = os.path.join(f"{algo}_{env_id}", "best_model.zip")

	# Load the saved model.
	model = algo.load(save_path, env=env)

	try:
		# Use deterministic actions for evaluation.
		episode_rewards, episode_lengths = [], []
		for _ in range(n_episodes):
			obs = env.reset()
			done = False
			episode_reward = 0.0
			episode_length = 0
			while not done:
				action, _ = model.predict(obs, deterministic=True)
				obs, reward, done, _info = env.step(action)
				episode_reward += reward

				episode_length += 1
				if not no_render:
					env.render(mode="human")
					dt = 1.0 / 240.0
					time.sleep(dt)
			episode_rewards.append(episode_reward)
			episode_lengths.append(episode_length)
			print(f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}.")

		mean_reward = np.mean(episode_rewards)
		std_reward = np.std(episode_rewards)

		mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)

		print("==== Results ====")
		print(f"Episode_reward = {mean_reward:.2f} +/- {std_reward:.2f}.")
		print(f"Episode_length = {mean_len:.2f} +/- {std_len:.2f}.")
	except KeyboardInterrupt:
		pass

	# Close process.
	env.close()

def main():
	# PyBullet.
	#	REF [site] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet
	# Examples.
	#	REF [site] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/examples

	# REF [site] >> https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym
	#	pybullet_data
	#	pybullet_envs
	#	pybullet_examples
	#	pybullet_robots
	#	pybullet_utils

	#hello_pybullet_world_introduction()

	#simulate_camera_images_test()
	#articulated_robot_test()
	ycb_benchmark_test()

	#--------------------
	# Environments.

	# Examples.
	#	python -m pybullet_envs.examples.enjoy_TF_HumanoidBulletEnv_v0_2017may
	#	python -m pybullet_envs.examples.racecarGymEnv
	#	python -m pybullet_envs.examples.kukaGymEnvTest

	#environment_test()

	#racecar_gym_env_example()
	#kuka_gym_env_example()

	#--------------------
	# Reinforcement learning.

	# Examples.
	#	python -m pybullet_envs.baselines.train_kuka_grasping
	#	python -m pybullet_envs.baselines.enjoy_kuka_grasping
	#	python -m pybullet_envs.baselines.train_pybullet_cartpole
	#	python -m pybullet_envs.baselines.enjoy_pybullet_cartpole
	#	python -m pybullet_envs.baselines.train_pybullet_racecar
	#	python -m pybullet_envs.baselines.enjoy_pybullet_racecar
	#	python -m pybullet_envs.stable_baselines.train
	#	python -m pybullet_envs.stable_baselines.enjoy

	# OpenAI Baselines.
	#baselines_train_kuka_grasping_example()
	#baselines_enjoy_kuka_grasping_example()
	#baselines_train_pybullet_cartpole_example()
	#baselines_enjoy_pybullet_cartpole_example()
	#baselines_train_pybullet_racecar_example()
	#baselines_enjoy_pybullet_racecar_example()

	# Stable Baselines3.
	#stable_baselines_train_example()
	#stable_baselines_enjoy_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
