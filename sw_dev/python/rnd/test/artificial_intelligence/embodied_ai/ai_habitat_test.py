#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import habitat

# REF [site] >> https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb
def habitat_sim_eccv_2020_navigation_tutorial():
	import sys, os, math, random
	import numpy as np
	from PIL import Image
	import habitat_sim
	from habitat_sim.utils import common as utils
	from habitat_sim.utils import viz_utils as vut
	from matplotlib import pyplot as plt

	show_video = False
	do_make_video = False
	display = False

	# Import the maps module alone for topdown mapping
	if display:
		from habitat.utils.visualizations import maps

	# Change to do something like this maybe: https://stackoverflow.com/a/41432704
	def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
		from habitat_sim.utils.common import d3_40_colors_rgb

		rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

		arr = [rgb_img]
		titles = ["rgb"]
		if semantic_obs.size != 0:
			semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
			semantic_img.putpalette(d3_40_colors_rgb.flatten())
			semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
			semantic_img = semantic_img.convert("RGBA")
			arr.append(semantic_img)
			titles.append("semantic")

		if depth_obs.size != 0:
			depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
			arr.append(depth_img)
			titles.append("depth")

		plt.figure(figsize=(12, 8))
		for i, data in enumerate(arr):
			ax = plt.subplot(1, 3, i + 1)
			ax.axis("off")
			ax.set_title(titles[i])
			plt.imshow(data)
		plt.show(block=False)

	# Basic settings.

	# This is the scene we are going to load.
	# We support a variety of mesh formats, such as .glb, .gltf, .obj, .ply.
	scene_file_path = "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"

	sim_settings = {
		"scene": scene_file_path,  # Scene path
		"default_agent": 0,  # Index of the default agent
		"sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
		"width": 256,  # Spatial resolution of the observations
		"height": 256,
	}

	# Configurations for the simulator.

	# This function generates a config for the simulator.
	# It contains two parts:
	# one for the simulator backend
	# one for the agent, where you can attach a bunch of sensors
	def make_simple_cfg(settings):
		# simulator backend
		sim_cfg = habitat_sim.SimulatorConfiguration()
		sim_cfg.scene_id = settings["scene"]

		# Agent
		agent_cfg = habitat_sim.agent.AgentConfiguration()

		# In the 1st example, we attach only one sensor,
		# a RGB visual sensor, to the agent
		rgb_sensor_spec = habitat_sim.CameraSensorSpec()
		rgb_sensor_spec.uuid = "color_sensor"
		rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
		rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
		rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

		agent_cfg.sensor_specifications = [rgb_sensor_spec]

		return habitat_sim.Configuration(sim_cfg, [agent_cfg])

	cfg = make_simple_cfg(sim_settings)

	# Create a simulator instance.
	try:  # Needed to handle out of order cell run in Colab
		sim.close()
	except NameError:
		pass
	sim = habitat_sim.Simulator(cfg)

	# Initialize an agent
	agent = sim.initialize_agent(sim_settings["default_agent"])

	# Set agent state
	agent_state = habitat_sim.AgentState()
	agent_state.position = np.array([-0.6, 0.0, 0.0])  # In world space
	agent.set_state(agent_state)

	# Get agent state
	agent_state = agent.get_state()
	print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

	# Navigate and see.

	# Obtain the default, discrete actions that an agent can perform
	# default action space contains 3 actions: move_forward, turn_left, and turn_right
	action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
	print("Discrete action space: ", action_names)

	def navigateAndSee(action=""):
		if action in action_names:
			observations = sim.step(action)
			print("action: ", action)
			if display:
				display_sample(observations["color_sensor"])

	action = "turn_right"
	navigateAndSee(action)

	action = "turn_right"
	navigateAndSee(action)

	action = "move_forward"
	navigateAndSee(action)

	action = "turn_left"
	navigateAndSee(action)

	#action = "move_backward"  # Illegal, no such action in the default action space
	#navigateAndSee(action)

# REF [site] >> https://github.com/facebookresearch/habitat-lab
def habitat_lab_example():
	import gym
	import habitat.gym

	if True:
		# Load embodied AI task (RearrangePick) and a pre-specified virtual robot.
		env = gym.make("HabitatRenderPick-v0")
	else:
		# To modify some of the configurations of the environment, you can also use the habitat.gym.make_gym_from_config method that allows you to create a habitat environment using a configuration.
		config = habitat.get_config(
			"benchmark/rearrange/pick.yaml",
			overrides=["habitat.environment.max_episode_steps=20"]
		)
		env = habitat.gym.make_gym_from_config(config)
	observations = env.reset()

	# Step through environment with random actions.
	terminal = False
	while not terminal:
		observations, reward, terminal, info = env.step(env.action_space.sample())

def main():
	# AI Habitat.
	#	PointGoal navigation.
	#		Go to [10, 0, -30].
	#	Object navigation.
	#		Go to toilet.
	#	Room navigation.
	#		Go to kitchen.
	#	Instruction following.
	#		Turn around, go out the bathroom, turn right, and go out the brown door.
	#	Embodied question answering.
	#		Q: What color is the TV stand? A: Dark blue.

	# Habitat-Sim.
	habitat_sim_eccv_2020_navigation_tutorial()  # Not yet completed.

	# Habitat-Lab.
	habitat_lab_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
