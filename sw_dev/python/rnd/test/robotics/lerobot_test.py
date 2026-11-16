#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/huggingface/lerobot/blob/main/examples/1_load_lerobot_dataset.py
def load_lerobot_dataset_example():
	from pprint import pprint
	import torch
	from huggingface_hub import HfApi
	import lerobot
	from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

	# We ported a number of existing datasets ourselves, use this to see the list:
	print("List of available datasets:")
	pprint(lerobot.available_datasets)

	# You can also browse through the datasets created/ported by the community on the hub using the hub api:
	hub_api = HfApi()
	repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
	pprint(repo_ids)

	# Or simply explore them in your web browser directly at:
	# https://huggingface.co/datasets?other=LeRobot

	# Let's take this one for this example
	repo_id = "lerobot/aloha_mobile_cabinet"
	# We can have a look and fetch its metadata to know more about it:
	ds_meta = LeRobotDatasetMetadata(repo_id)

	# By instantiating just this class, you can quickly access useful information about the content and the
	# structure of the dataset without downloading the actual data yet (only metadata files — which are
	# lightweight).
	print(f"Total number of episodes: {ds_meta.total_episodes}")
	print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
	print(f"Frames per second used during data collection: {ds_meta.fps}")
	print(f"Robot type: {ds_meta.robot_type}")
	print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

	print("Tasks:")
	print(ds_meta.tasks)
	print("Features:")
	pprint(ds_meta.features)

	# You can also get a short summary by simply printing the object:
	print(ds_meta)

	# You can then load the actual dataset from the hub.
	# Either load any subset of episodes:
	dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])

	# And see how many frames you have:
	print(f"Selected episodes: {dataset.episodes}")
	print(f"Number of episodes selected: {dataset.num_episodes}")
	print(f"Number of frames selected: {dataset.num_frames}")

	# Or simply load the entire dataset:
	dataset = LeRobotDataset(repo_id)
	print(f"Number of episodes selected: {dataset.num_episodes}")
	print(f"Number of frames selected: {dataset.num_frames}")

	# The previous metadata class is contained in the 'meta' attribute of the dataset:
	print(dataset.meta)

	# LeRobotDataset actually wraps an underlying Hugging Face dataset
	# (see https://huggingface.co/docs/datasets for more information).
	print(dataset.hf_dataset)

	# LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
	# with the latter, like iterating through the dataset.
	# The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
	# episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
	# frame indices associated to the first episode:
	episode_index = 0
	from_idx = dataset.episode_data_index["from"][episode_index].item()
	to_idx = dataset.episode_data_index["to"][episode_index].item()

	# Then we grab all the image frames from the first camera:
	camera_key = dataset.meta.camera_keys[0]
	frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

	# The objects returned by the dataset are all torch.Tensors
	print(type(frames[0]))
	print(frames[0].shape)

	# Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
	# We can compare this shape with the information available for that feature
	pprint(dataset.features[camera_key])
	# In particular:
	print(dataset.features[camera_key]["shape"])
	# The shape is in (h, w, c) which is a more universal format.

	# For many machine learning applications we need to load the history of past observations or trajectories of
	# future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
	# differences with the current loaded frame. For instance:
	delta_timestamps = {
		# loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
		camera_key: [-1, -0.5, -0.20, 0],
		# loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
		"observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
		# loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
		"action": [t / dataset.fps for t in range(64)],
	}
	# Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
	# timestamp, you still get a valid timestamp.

	dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
	print(f"\n{dataset[0][camera_key].shape=}")  # (4, c, h, w)
	print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
	print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

	# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
	# PyTorch datasets.
	dataloader = torch.utils.data.DataLoader(
		dataset,
		num_workers=0,
		batch_size=32,
		shuffle=True,
	)

	for batch in dataloader:
		print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
		print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
		print(f"{batch['action'].shape=}")  # (32, 64, c)
		break

# REF [site] >> https://github.com/huggingface/lerobot/blob/main/examples/2_evaluate_pretrained_policy.py
def evaluate_pretrained_policy_example():
	from pathlib import Path
	import numpy
	import imageio
	import torch
	import gym_pusht  # noqa: F401
	import gymnasium as gym
	from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

	# Create a directory to store the video of the evaluation
	output_directory = Path("outputs/eval/example_pusht_diffusion")
	output_directory.mkdir(parents=True, exist_ok=True)

	# Select your device
	device = "cuda"

	# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
	pretrained_policy_path = "lerobot/diffusion_pusht"
	# OR a path to a local outputs/train folder.
	# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

	policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

	# Initialize evaluation environment to render two observation types:
	# an image of the scene and state/position of the agent. The environment
	# also automatically stops running after 300 interactions/steps.
	env = gym.make(
		"gym_pusht/PushT-v0",
		obs_type="pixels_agent_pos",
		max_episode_steps=300,
	)

	# We can verify that the shapes of the features expected by the policy match the ones from the observations
	# produced by the environment
	print(policy.config.input_features)
	print(env.observation_space)

	# Similarly, we can check that the actions produced by the policy will match the actions expected by the
	# environment
	print(policy.config.output_features)
	print(env.action_space)

	# Reset the policy and environments to prepare for rollout
	policy.reset()
	numpy_observation, info = env.reset(seed=42)

	# Prepare to collect every rewards and all the frames of the episode,
	# from initial state to final state.
	rewards = []
	frames = []

	# Render frame of the initial state
	frames.append(env.render())

	step = 0
	done = False
	while not done:
		# Prepare observation for the policy running in Pytorch
		state = torch.from_numpy(numpy_observation["agent_pos"])
		image = torch.from_numpy(numpy_observation["pixels"])

		# Convert to float32 with image from channel first in [0,255]
		# to channel last in [0,1]
		state = state.to(torch.float32)
		image = image.to(torch.float32) / 255
		image = image.permute(2, 0, 1)

		# Send data tensors from CPU to GPU
		state = state.to(device, non_blocking=True)
		image = image.to(device, non_blocking=True)

		# Add extra (empty) batch dimension, required to forward the policy
		state = state.unsqueeze(0)
		image = image.unsqueeze(0)

		# Create the policy input dictionary
		observation = {
			"observation.state": state,
			"observation.image": image,
		}

		# Predict the next action with respect to the current observation
		with torch.inference_mode():
			action = policy.select_action(observation)

		# Prepare the action for the environment
		numpy_action = action.squeeze(0).to("cpu").numpy()

		# Step through the environment and receive a new observation
		numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
		print(f"{step=} {reward=} {terminated=}")

		# Keep track of all the rewards and frames
		rewards.append(reward)
		frames.append(env.render())

		# The rollout is considered done when the success state is reached (i.e. terminated is True),
		# or the maximum number of iterations is reached (i.e. truncated is True)
		done = terminated | truncated | done
		step += 1

	if terminated:
		print("Success!")
	else:
		print("Failure!")

	# Get the speed of environment (i.e. its number of frames per second).
	fps = env.metadata["render_fps"]

	# Encode all frames into a mp4 video.
	video_path = output_directory / "rollout.mp4"
	imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

	print(f"Video of the evaluation is available in '{video_path}'.")

# REF [site] >> https://github.com/huggingface/lerobot/blob/main/examples/3_train_policy.py
def train_policy_example():
	from pathlib import Path
	import torch
	from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
	from lerobot.common.datasets.utils import dataset_to_policy_features
	from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
	from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
	from lerobot.configs.types import FeatureType

	# Create a directory to store the training checkpoint.
	output_directory = Path("outputs/train/example_pusht_diffusion")
	output_directory.mkdir(parents=True, exist_ok=True)

	# Select your device
	device = torch.device("cuda")

	# Number of offline training steps (we'll only do offline training for this example.)
	# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
	training_steps = 5000
	log_freq = 1

	# When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
	# creating the policy:
	#	- input/output shapes: to properly size the policy
	#	- dataset stats: for normalization and denormalization of input/outputs
	dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
	features = dataset_to_policy_features(dataset_metadata.features)
	output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
	input_features = {key: ft for key, ft in features.items() if key not in output_features}

	# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
	# we'll just use the defaults and so no arguments other than input/output features need to be passed.
	cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

	# We can now instantiate our policy with this config and the dataset stats.
	policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
	policy.train()
	policy.to(device)

	# Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
	# which can differ for inputs, outputs and rewards (if there are some).
	delta_timestamps = {
		"observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
		"observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
		"action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
	}

	# In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
	delta_timestamps = {
		# Load the previous image and state at -0.1 seconds before current frame,
		# then load current image and state corresponding to 0.0 second.
		"observation.image": [-0.1, 0.0],
		"observation.state": [-0.1, 0.0],
		# Load the previous action (-0.1), the next action to be executed (0.0),
		# and 14 future actions with a 0.1 seconds spacing. All these actions will be
		# used to supervise the policy.
		"action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
	}

	# We can then instantiate the dataset with these delta_timestamps configuration.
	dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

	# Then we create our optimizer and dataloader for offline training.
	optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		num_workers=4,
		batch_size=64,
		shuffle=True,
		pin_memory=device.type != "cpu",
		drop_last=True,
	)

	# Run training loop.
	step = 0
	done = False
	while not done:
		for batch in dataloader:
			batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
			loss, _ = policy.forward(batch)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			if step % log_freq == 0:
				print(f"step: {step} loss: {loss.item():.3f}")
			step += 1
			if step >= training_steps:
				done = True
				break

	# Save a policy checkpoint.
	policy.save_pretrained(output_directory)

def main():
	# References
	#	https://github.com/huggingface/lerobot
	#	https://huggingface.co/lerobot

	# Install:
	#	git clone https://github.com/huggingface/lerobot.git
	#	pip install -e .
	#	pip install -e ".[aloha, pusht]"

	load_lerobot_dataset_example()
	#evaluate_pretrained_policy_example()
	#train_policy_example()

	# pi0
	#	REF [function] >> lerobot_pi0_test() in ./vla_test.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
