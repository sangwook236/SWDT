#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
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

	for i in range(10000):
		p.stepSimulation()

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
		total = sum(lcl['episode_rewards'][-101:-1]) / 100
		totalt = lcl['t']
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
		is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
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

	#env = gym.make('CartPoleBulletEnv-v1')
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
		total = sum(lcl['episode_rewards'][-101:-1]) / 100
		totalt = lcl['t']
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
		algo = SAC  # RL Algorithm.
		algo_id = "sac"
		# Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
		hyperparams = dict(
			batch_size=256,
			gamma=0.98,
			policy_kwargs=dict(net_arch=[256, 256]),
			learning_starts=10000,
			buffer_size=int(3e5),
			tau=0.01,
		)
	elif False:
		algo = TD3  # RL Algorithm.
		algo_id = "td3"
		# Tuned hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
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
	stable_baselines_train_example()
	#stable_baselines_enjoy_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
