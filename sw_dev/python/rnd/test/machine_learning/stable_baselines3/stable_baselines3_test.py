#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC, DDPG, TD3, HerReplayBuffer
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize, VecExtractDictObs, VecMonitor, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
def getting_started():
	import gym

	env = gym.make("CartPole-v1")
	model = A2C("MlpPolicy", env, verbose=1)
	#model = A2C("MlpPolicy", "CartPole-v1", verbose=1)

	model.learn(total_timesteps=10000)

	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			obs = env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def basic_usage_example():
	import gym

	# Basic Usage: Training, Saving, Loading.

	# Create environment.
	env = gym.make("LunarLander-v2")

	# Instantiate the agent.
	model = DQN("MlpPolicy", env, verbose=1)
	# Train the agent.
	model.learn(total_timesteps=int(2e5))
	# Save the agent.
	model.save("dqn_lunar")
	del model  # Delete trained model to demonstrate loading.

	# Load the trained agent.
	# NOTE: if you have loading issue, you can pass 'print_system_info=True'
	# to compare the system on which the model was trained vs the current one.
	#model = DQN.load("dqn_lunar", env=env, print_system_info=True)
	model = DQN.load("dqn_lunar", env=env)

	# Evaluate the agent.
	# NOTE: If you use wrappers with your environment that modify rewards,
	#	this will be reflected here. To evaluate with original rewards,
	#	wrap environment in a "Monitor" wrapper before other wrappers.
	mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

	# Enjoy trained agent.
	obs = env.reset()
	for i in range(1000):
		action, states = model.predict(obs, deterministic=True)
		obs, rewards, dones, info = env.step(action)
		env.render()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def multiprocessing_example():
	import gym

	# Multiprocessing: Unleashing the Power of Vectorized Environments

	def make_env(env_id, rank, seed=0):
		"""
		Utility function for multiprocessed env.

		:param env_id: (str) the environment ID.
		:param num_env: (int) the number of environments you wish to have in subprocesses.
		:param seed: (int) the inital seed for RNG.
		:param rank: (int) index of the subprocess.
		"""
		def _init():
			env = gym.make(env_id)
			env.seed(seed + rank)
			return env
		set_random_seed(seed)
		return _init

	env_id = "CartPole-v1"
	num_cpu = 4  # Number of processes to use.
	# Create the vectorized environment.
	env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

	# Stable Baselines provides you with make_vec_env() helper which does exactly the previous steps for you.
	# You can choose between 'DummyVecEnv' (usually faster) and 'SubprocVecEnv'.
	#env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=25_000)

	obs = env.reset()
	for _ in range(1000):
		action, states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def multiprocessing_with_off_policy_algorithms_example():
	# Multiprocessing with off-policy algorithms.

	env = make_vec_env("Pendulum-v1", n_envs=4, seed=0)

	# We collect 4 transitions per call to 'env.step()' and performs 2 gradient steps per call to 'env.step()'
	# if gradient_steps=-1, then we would do 4 gradients steps per call to 'env.step()'.
	model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)
	model.learn(total_timesteps=10_000)

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def using_callback_example():
	import gym

	# Using Callback: Monitoring Training.

	class SaveOnBestTrainingRewardCallback(BaseCallback):
		"""
		Callback for saving a model (the check is done every 'check_freq' steps)
		based on the training reward (in practice, we recommend using 'EvalCallback').

		:param check_freq:
		:param log_dir: Path to the folder where the model will be saved. It must contains the file created by the 'Monitor' wrapper.
		:param verbose: Verbosity level.
		"""
		def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
			super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
			self.check_freq = check_freq
			self.log_dir = log_dir
			self.save_path = os.path.join(log_dir, "best_model")
			self.best_mean_reward = -np.inf

		def _init_callback(self) -> None:
			# Create folder if needed.
			if self.save_path is not None:
				os.makedirs(self.save_path, exist_ok=True)

		def _on_step(self) -> bool:
			if self.n_calls % self.check_freq == 0:
				# Retrieve training reward.
				x, y = ts2xy(load_results(self.log_dir), "timesteps")
				if len(x) > 0:
					# Mean training reward over the last 100 episodes.
					mean_reward = np.mean(y[-100:])
					if self.verbose > 0:
						print(f"Num timesteps: {self.num_timesteps}")
						print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

					# New best model, you could save the agent here.
					if mean_reward > self.best_mean_reward:
						self.best_mean_reward = mean_reward
						# Example for saving best model.
						if self.verbose > 0:
							print(f"Saving new best model to {self.save_path}")
						self.model.save(self.save_path)

			return True

	# Create log dir.
	log_dir = "tmp/"
	os.makedirs(log_dir, exist_ok=True)

	# Create and wrap the environment.
	env = gym.make("LunarLanderContinuous-v2")
	env = Monitor(env, log_dir)

	# Add some action noise for exploration.
	n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
	# Because we use parameter noise, we should use a MlpPolicy with layer normalization.
	model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0)
	# Create the callback: check every 1000 steps.
	callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
	# Train the agent.
	timesteps = 1e5
	model.learn(total_timesteps=int(timesteps), callback=callback)

	plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
	plt.show()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def atari_games_example():
	# There already exists an environment generator that will make and wrap atari environments correctly.
	# Here we are also multi-worker training (n_envs=4 => 4 environments).
	env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
	# Frame-stacking with 4 frames.
	env = VecFrameStack(env, n_stack=4)

	model = A2C("CnnPolicy", env, verbose=1)
	model.learn(total_timesteps=25_000)

	obs = env.reset()
	while True:
		action, states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		env.render()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def pybullet_example():
	import gym
	import pybullet_envs

	# PyBullet: Normalizing input features.

	env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
	# Automatically normalize the input features and reward.
	env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

	model = PPO("MlpPolicy", env)
	model.learn(total_timesteps=2000)

	# Don't forget to save the VecNormalize statistics when saving the agent.
	log_dir = "/tmp/"
	model.save(log_dir + "ppo_halfcheetah")
	stats_path = os.path.join(log_dir, "vec_normalize.pkl")
	env.save(stats_path)

	# To demonstrate loading.
	del model, env

	# Load the saved statistics.
	env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
	env = VecNormalize.load(stats_path, env)
	# Do not update them at test time.
	env.training = False
	# reward normalization is not needed at test time.
	env.norm_reward = False

	# Load the agent.
	model = PPO.load(log_dir + "ppo_halfcheetah", env=env)

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def hindsight_experience_replay_example():
	import gym
	import highway_env

	# Hindsight Experience Replay (HER).

	env = gym.make("parking-v0")

	# Create 4 artificial transitions per real transition.
	n_sampled_goal = 4

	# SAC hyperparams:
	model = SAC(
		"MultiInputPolicy",
		env,
		replay_buffer_class=HerReplayBuffer,
		replay_buffer_kwargs=dict(
			n_sampled_goal=n_sampled_goal,
			goal_selection_strategy="future",
			# IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
			# we have to manually specify the max number of steps per episode.
			max_episode_length=100,
			online_sampling=True,
		),
		verbose=1,
		buffer_size=int(1e6),
		learning_rate=1e-3,
		gamma=0.95,
		batch_size=256,
		policy_kwargs=dict(net_arch=[256, 256, 256]),
	)

	model.learn(int(2e5))
	model.save("her_sac_highway")

	# Load saved model.
	# Because it needs access to 'env.compute_reward()'
	# HER must be loaded with the env.
	model = SAC.load("her_sac_highway", env=env)

	obs = env.reset()

	# Evaluate the agent.
	episode_reward = 0
	for _ in range(100):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		env.render()
		episode_reward += reward
		if done or info.get("is_success", False):
			print("Reward:", episode_reward, "Success?", info.get("is_success", False))
			episode_reward = 0.0
			obs = env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def learning_rate_schedule_example():
	from typing import Callable

	# Learning Rate Schedule.

	def linear_schedule(initial_value: float) -> Callable[[float], float]:
		"""
		Linear learning rate schedule.

		:param initial_value: Initial learning rate.
		:return: schedule that computes current learning rate depending on remaining progress.
		"""
		def func(progress_remaining: float) -> float:
			"""
			Progress will decrease from 1 (beginning) to 0.

			:param progress_remaining:
			:return: current learning rate.
			"""
			return progress_remaining * initial_value

		return func

	# Initial learning rate of 0.001.
	model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
	model.learn(total_timesteps=20_000)
	# By default, 'reset_num_timesteps' is True, in which case the learning rate schedule resets.
	#progress_remaining = 1.0 - (num_timesteps / total_timesteps)
	model.learn(total_timesteps=10_000, reset_num_timesteps=True)

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def advanced_saving_and_loading_example():
	from stable_baselines3.sac.policies import MlpPolicy

	# Advanced Saving and Loading.

	# Create the model, the training environment and the test environment (for evaluation).
	model = SAC("MlpPolicy", "Pendulum-v1", verbose=1, learning_rate=1e-3, create_eval_env=True)

	# Evaluate the model every 1000 steps on 5 test episodes and save the evaluation to the "logs/" folder.
	model.learn(6000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")

	# Save the model.
	model.save("sac_pendulum")

	# The saved model does not contain the replay buffer.
	loaded_model = SAC.load("sac_pendulum")
	print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

	# Now save the replay buffer too.
	model.save_replay_buffer("sac_replay_buffer")

	# Load it into the loaded_model.
	loaded_model.load_replay_buffer("sac_replay_buffer")

	# Now the loaded replay is not empty anymore.
	print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

	# Save the policy independently from the model.
	# Note: if you don't save the complete model with 'model.save()'
	# you cannot continue training afterward.
	policy = model.policy
	policy.save("sac_policy_pendulum")

	# Retrieve the environment.
	env = model.get_env()

	# Evaluate the policy.
	mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

	print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

	# Load the policy independently from the model.
	saved_policy = MlpPolicy.load("sac_policy_pendulum")

	# Evaluate the loaded policy.
	mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)

	print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def accessing_and_modifying_model_parameters_example():
	from typing import Dict
	import torch

	# Accessing and modifying model parameters.

	def mutate(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		"""Mutate parameters by adding normal noise to them"""
		return dict((name, param + torch.randn_like(param)) for name, param in params.items())

	# Create policy with a small network.
	model = A2C(
		"MlpPolicy",
		"CartPole-v1",
		ent_coef=0.0,
		policy_kwargs={"net_arch": [32]},
		seed=0,
		learning_rate=0.05,
	)

	# Use traditional actor-critic policy gradient updates to find good initial parameters.
	model.learn(total_timesteps=10_000)

	# Include only variables with "policy", "action" (policy) or "shared_net" (shared layers) in their name: only these ones affect the action.
	# NOTE: you can retrieve those parameters using model.get_parameters() too.
	mean_params = dict((key, value) for key, value in model.policy.state_dict().items() if ("policy" in key or "shared_net" in key or "action" in key))

	# Population size of 50 invdiduals.
	pop_size = 50
	# Keep top 10%.
	n_elite = pop_size // 10
	# Retrieve the environment.
	env = model.get_env()

	for iteration in range(10):
		# Create population of candidates and evaluate them.
		population = []
		for population_i in range(pop_size):
			candidate = mutate(mean_params)
			# Load new policy parameters to agent.
			# Tell function that it should only update parameters we give it (policy parameters).
			model.policy.load_state_dict(candidate, strict=False)
			# Evaluate the candidate.
			fitness, _ = evaluate_policy(model, env)
			population.append((candidate, fitness))
		# Take top 10% and use average over their parameters as next mean parameter.
		top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:n_elite]
		mean_params = dict((name, torch.stack([candidate[0][name] for candidate in top_candidates]).mean(dim=0),) for name in mean_params.keys())
		mean_fitness = sum(top_candidate[1] for top_candidate in top_candidates) / n_elite
		print(f"Iteration {iteration + 1:<3} Mean top fitness: {mean_fitness:.2f}")
		print(f"Best fitness: {top_candidates[0][1]:.2f}")

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def SB3_and_ProcgenEnv_example():
	# SB3 and ProcgenEnv.

	from procgen import ProcgenEnv

	# ProcgenEnv is already vectorized.
	venv = ProcgenEnv(num_envs=2, env_name="starpilot")

	# To use only part of the observation:
	#venv = VecExtractDictObs(venv, "rgb")

	# Wrap with a VecMonitor to collect stats and avoid errors.
	venv = VecMonitor(venv=venv)

	model = PPO("MultiInputPolicy", venv, verbose=1)
	model.learn(10_000)

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def record_video_example():
	import gym

	# Record a Video.

	env_id = "CartPole-v1"
	video_folder = "logs/videos/"
	video_length = 100

	env = DummyVecEnv([lambda: gym.make(env_id)])

	obs = env.reset()

	# Record the video starting at the first step.
	env = VecVideoRecorder(
		env, video_folder,
		record_video_trigger=lambda x: x == 0, video_length=video_length,
		name_prefix=f"random-agent-{env_id}"
	)

	env.reset()
	for _ in range(video_length + 1):
		action = [env.action_space.sample()]
		obs, _, _, _ = env.step(action)

	# Save the video.
	env.close()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
def make_gif_example():
	import imageio

	# Make a GIF of a Trained Agent.

	model = A2C("MlpPolicy", "LunarLander-v2").learn(100_000)

	images = []
	obs = model.env.reset()
	img = model.env.render(mode="rgb_array")
	for i in range(350):
		images.append(img)
		action, _ = model.predict(obs)
		obs, _, _ ,_ = model.env.step(action)
		img = model.env.render(mode="rgb_array")

	imageio.mimsave("./lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
def dqn_module_example():
	from stable_baselines3 import DQN
	import gymnasium as gym

	env = gym.make("CartPole-v1", render_mode="human")
	#env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
	print(f"Max epsode steps = {env.spec.max_episode_steps}.")

	model = DQN("MlpPolicy", env, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=10000, log_interval=4)
	print(f"Trained: {time.time() - start_time} secs.")

	model.save("./dqn_cartpole")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = DQN.load("./dqn_cartpole")

	obs, info = env.reset()
	while True:
		action, states = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)  # numpy.ndarray, float, bool, bool, dict
		if terminated or truncated:
			print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}, Info: {info}.")
			obs, info = env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
def a2c_module_example():
	from stable_baselines3 import A2C
	from stable_baselines3.common.env_util import make_vec_env

	# Parallel environments
	vec_env = make_vec_env("CartPole-v1", n_envs=4)
	model = A2C("MlpPolicy", vec_env, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=25000)
	print(f"Trained: {time.time() - start_time} secs.")

	model.save("./a2c_cartpole")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = A2C.load("./a2c_cartpole")

	obs = vec_env.reset()
	while True:
		action, states = model.predict(obs)
		obs, rewards, dones, infos = vec_env.step(action)  # numpy.ndarray, numpy.ndarray, numpy.ndarray, list
		vec_env.render("human")
		if dones.any():
			print(f"Dones: {dones}, Rewards: {rewards}, Infos: {infos}.")  # infos: [{"episode": {"r": <cumulative reward>, "l": <episode length>, "t": <elapsed time since instantiation of wrapper>}, ...}]
			#obs = vec_env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
def ppo_module_example():
	from stable_baselines3 import PPO
	from stable_baselines3.common.env_util import make_vec_env

	# Parallel environments
	vec_env = make_vec_env("CartPole-v1", n_envs=4)
	model = PPO("MlpPolicy", vec_env, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=250000)
	print(f"Trained: {time.time() - start_time} secs.")

	model.save("./ppo_cartpole")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = PPO.load("./ppo_cartpole")

	obs = vec_env.reset()
	while True:
		action, states = model.predict(obs)
		obs, rewards, dones, infos = vec_env.step(action)  # numpy.ndarray, numpy.ndarray, numpy.ndarray, list
		vec_env.render("human")
		if dones.any():
			print(f"Dones: {dones}, Rewards: {rewards}, Infos: {infos}.")  # infos: [{"episode": {"r": <cumulative reward>, "l": <episode length>, "t": <elapsed time since instantiation of wrapper>}, ...}]
			#obs = vec_env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
def ddpg_module_example():
	import numpy as np
	from stable_baselines3 import DDPG
	from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
	import gymnasium as gym

	env = gym.make("Pendulum-v1", render_mode="rgb_array")
	#env = gym.make("Pendulum-v1", render_mode="rgb_array", max_episode_steps=500)
	print(f"Max epsode steps = {env.spec.max_episode_steps}.")

	# The noise objects for DDPG
	n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
	model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=10000, log_interval=10)
	print(f"Trained: {time.time() - start_time} secs.")

	vec_env = model.get_env()

	model.save("./ddpg_pendulum")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = DDPG.load("./ddpg_pendulum")

	obs = vec_env.reset()
	while True:
		action, states = model.predict(obs)
		obs, rewards, dones, infos = vec_env.step(action)  # numpy.ndarray, numpy.ndarray, numpy.ndarray, list
		vec_env.render("human")
		if dones.any():
			print(f"Dones: {dones}, Rewards: {rewards}, Infos: {infos}.")  # infos: [{"episode": {"r": <cumulative reward>, "l": <episode length>, "t": <elapsed time since instantiation of wrapper>}, ...}]
			#obs = vec_env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
def td3_module_example():
	import numpy as np
	from stable_baselines3 import TD3
	from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
	import gymnasium as gym

	env = gym.make("Pendulum-v1", render_mode="rgb_array")
	#env = gym.make("Pendulum-v1", render_mode="rgb_array", max_episode_steps=500)
	print(f"Max epsode steps = {env.spec.max_episode_steps}.")

	# The noise objects for TD3
	n_actions = env.action_space.shape[-1]
	action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
	model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=10000, log_interval=10)
	print(f"Trained: {time.time() - start_time} secs.")

	vec_env = model.get_env()

	model.save("./td3_pendulum")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = TD3.load("./td3_pendulum")

	obs = vec_env.reset()
	while True:
		action, states = model.predict(obs)
		obs, rewards, dones, infos = vec_env.step(action)  # numpy.ndarray, numpy.ndarray, numpy.ndarray, list
		vec_env.render("human")
		if dones.any():
			print(f"Dones: {dones}, Rewards: {rewards}, Infos: {infos}.")  # infos: [{"episode": {"r": <cumulative reward>, "l": <episode length>, "t": <elapsed time since instantiation of wrapper>}, ...}]
			#obs = vec_env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
def sac_module_example():
	from stable_baselines3 import SAC
	import gymnasium as gym

	env = gym.make("Pendulum-v1", render_mode="human")
	#env = gym.make("Pendulum-v1", render_mode="human", max_episode_steps=500)
	print(f"Max epsode steps = {env.spec.max_episode_steps}.")

	model = SAC("MlpPolicy", env, verbose=1)

	print("Training...")
	start_time = time.time()
	model.learn(total_timesteps=10000, log_interval=4)
	print(f"Trained: {time.time() - start_time} secs.")

	model.save("./sac_pendulum")
	del model  # Remove to demonstrate saving and loading

	#-----
	model = SAC.load("./sac_pendulum")

	obs, info = env.reset()
	while True:
		action, states = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)  # numpy.ndarray, numpy.float64, bool, bool, dict
		if terminated or truncated:
			print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}, Info: {info}.")
			obs, info = env.reset()

# REF [site] >> https://stable-baselines3.readthedocs.io/en/master/modules/her.html
def her_module_example():
	from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
	from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
	from stable_baselines3.common.envs import BitFlippingEnv

	model_class = DQN  # Works also with SAC, DDPG and TD3
	N_BITS = 15

	env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

	# Available strategies (cf. paper): future, final, episode
	goal_selection_strategy = "future"  # Equivalent to GoalSelectionStrategy.FUTURE

	# Initialize the model
	model = model_class(
		"MultiInputPolicy",
		env,
		replay_buffer_class=HerReplayBuffer,
		# Parameters for HER
		replay_buffer_kwargs=dict(
			n_sampled_goal=4,
			goal_selection_strategy=goal_selection_strategy,
		),
		verbose=1,
	)

	# Train the model
	print("Training...")
	start_time = time.time()
	model.learn(1000)
	print(f"Trained: {time.time() - start_time} secs.")

	model.save("./her_bit_env")
	del model  # Remove to demonstrate saving and loading

	#-----
	# Because it needs access to `env.compute_reward()`, HER must be loaded with the env
	model = model_class.load("./her_bit_env", env=env)

	obs, info = env.reset()
	for _ in range(100):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, info = env.step(action)
		if terminated or truncated:
			print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}, Info: {info}.")
			obs, info = env.reset()

def main():
	#getting_started()

	#basic_usage_example()
	#multiprocessing_example()
	#multiprocessing_with_off_policy_algorithms_example()
	#using_callback_example()

	#atari_games_example()
	#pybullet_example()

	#hindsight_experience_replay_example()
	#learning_rate_schedule_example()

	#advanced_saving_and_loading_example()
	#accessing_and_modifying_model_parameters_example()
	#SB3_and_ProcgenEnv_example()
	#record_video_example()
	#make_gif_example()

	#-----
	# Value-function-based algorithm

	# Deep Q network (DQN)
	#dqn_module_example()

	#-----
	# Policy gradient algorithm

	# Advantage actor critic (A2C)
	#	A synchronous, deterministic variant of asynchronous advantage actor critic (A3C)
	#	It uses multiple workers to avoid the use of a replay buffer
	#a2c_module_example()
	# Proximal policy optimization (PPO)
	#	It combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor)
	#ppo_module_example()
	# Deep deterministic policy gradient (DDPG)
	#	Model-free, off-policy actor-critic algorithm
	#	It combines the trick for DQN with the deterministic policy gradient, to obtain an algorithm for continuous actions
	#ddpg_module_example()
	# Twin delayed DDPG (TD3)
	#	Function approximation error in actor-critic methods
	#	TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing
	td3_module_example()
	# Soft actor critic (SAC)
	#	Off-policy maximum entropy deep reinforcement learning with a stochastic actor
	#sac_module_example()

	#-----

	# Hindsight experience replay (HER)
	#her_module_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
