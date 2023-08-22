#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def dqn_cart_pole_tutorial():
	import gym
	import math
	import random
	import numpy as np
	import matplotlib
	import matplotlib.pyplot as plt
	from collections import namedtuple, deque
	from itertools import count
	from PIL import Image

	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torch.nn.functional as F
	import torchvision.transforms as T

	# REF [site] >> https://www.gymlibrary.dev/environments/classic_control/cart_pole/
	env = gym.make("CartPole-v0").unwrapped

	# Set up matplotlib.
	is_ipython = "inline" in matplotlib.get_backend()
	if is_ipython:
		from IPython import display

	plt.ion()

	# If gpu is to be used.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# Replay memory.
	Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

	class ReplayMemory(object):
		def __init__(self, capacity):
			self.memory = deque([], maxlen=capacity)

		def push(self, *args):
			"""Save a transition"""
			self.memory.append(Transition(*args))

		def sample(self, batch_size):
			return random.sample(self.memory, batch_size)

		def __len__(self):
			return len(self.memory)

	# DQN algorithm.
	class DQN(nn.Module):
		def __init__(self, h, w, outputs):
			super(DQN, self).__init__()
			self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
			self.bn3 = nn.BatchNorm2d(32)

			# Number of Linear input connections depends on output of conv2d layers and therefore the input image size, so compute it.
			def conv2d_size_out(size, kernel_size=5, stride=2):
				return (size - (kernel_size - 1) - 1) // stride + 1
			convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
			convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
			linear_input_size = convw * convh * 32
			self.head = nn.Linear(linear_input_size, outputs)

		# Called with either one element to determine next action, or a batch during optimization.
		# Returns tensor([[left0exp, right0exp]...]).
		def forward(self, x):
			x = x.to(device)
			x = F.relu(self.bn1(self.conv1(x)))
			x = F.relu(self.bn2(self.conv2(x)))
			x = F.relu(self.bn3(self.conv3(x)))
			return self.head(x.view(x.size(0), -1))  # Expected values, but not actions. Actions can be chosen from the expected values.

	# Input extraction.
	resize = T.Compose([
		T.ToPILImage(),
		T.Resize(40, interpolation=Image.CUBIC),
		T.ToTensor()
	])

	def get_cart_location(screen_width):
		world_width = env.x_threshold * 2
		scale = screen_width / world_width
		return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART.

	def get_screen():
		# Returned screen requested by gym is 400x600x3, but is sometimes larger such as 800x1200x3. Transpose it into torch order (CHW).
		screen = env.render(mode="rgb_array").transpose((2, 0, 1))
		# Cart is in the lower half, so strip off the top and bottom of the screen.
		_, screen_height, screen_width = screen.shape
		screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
		view_width = int(screen_width * 0.6)
		cart_location = get_cart_location(screen_width)
		if cart_location < view_width // 2:
			slice_range = slice(view_width)
		elif cart_location > (screen_width - view_width // 2):
			slice_range = slice(-view_width, None)
		else:
			slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
		# Strip off the edges, so that we have a square image centered on a cart.
		screen = screen[:, :, slice_range]
		# Convert to float, rescale, convert to torch tensor (this doesn't require a copy).
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)
		# Resize, and add a batch dimension (BCHW).
		return resize(screen).unsqueeze(0)

	env.reset()
	plt.figure()
	plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation="none")
	plt.title("Example extracted screen")
	plt.show()

	# Training.
	BATCH_SIZE = 128
	GAMMA = 0.999
	EPS_START = 0.9
	EPS_END = 0.05
	EPS_DECAY = 200
	TARGET_UPDATE = 10

	# Get screen size so that we can initialize layers correctly based on shape returned from AI gym.
	# Typical dimensions at this point are close to 3x40x90 which is the result of a clamped and down-scaled render buffer in get_screen().
	init_screen = get_screen()
	_, _, screen_height, screen_width = init_screen.shape

	# Get number of actions from gym action space.
	n_actions = env.action_space.n

	policy_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net = DQN(screen_height, screen_width, n_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	optimizer = optim.RMSprop(policy_net.parameters())
	memory = ReplayMemory(10000)

	steps_done = 0

	def select_action(state, steps_done):
		#global steps_done
		sample = random.random()
		eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
		steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
				# t.max(1) will return largest column value of each row.
				# Second column on max result is index of where max element was found, so we pick action with the larger expected reward.
				return policy_net(state).max(1)[1].view(1, 1), steps_done
		else:
			return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done

	episode_durations = []

	def plot_durations():
		plt.figure(2)
		plt.clf()
		durations_t = torch.tensor(episode_durations, dtype=torch.float)
		plt.title("Training...")
		plt.xlabel("Episode")
		plt.ylabel("Duration")
		plt.plot(durations_t.numpy())
		# Take 100 episode averages and plot them too.
		if len(durations_t) >= 100:
			means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.zeros(99), means))
			plt.plot(means.numpy())

		plt.pause(0.001)  # Pause a bit so that plots are updated.
		if is_ipython:
			display.clear_output(wait=True)
			display.display(plt.gcf())

	def optimize_model():
		if len(memory) < BATCH_SIZE:
			return
		transitions = memory.sample(BATCH_SIZE)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
		# This converts batch-array of Transitions to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended).
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
		# These are the actions which would've been taken for each batch state according to policy_net.
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
		next_state_values = torch.zeros(BATCH_SIZE, device=device)
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
		# Compute the expected Q values.
		expected_state_action_values = (next_state_values * GAMMA) + reward_batch

		# Compute Huber loss.
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model.
		optimizer.zero_grad()
		loss.backward()
		for param in policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()

	num_episodes = 50
	for i_episode in range(num_episodes):
		# Initialize the environment and state.
		env.reset()
		last_screen = get_screen()
		current_screen = get_screen()
		state = current_screen - last_screen
		for t in count():
			# Select and perform an action.
			action, steps_done = select_action(state, steps_done)
			_, reward, done, _ = env.step(action.item())
			reward = torch.tensor([reward], device=device)

			# Observe new state.
			last_screen = current_screen
			current_screen = get_screen()
			if not done:
				next_state = current_screen - last_screen
			else:
				next_state = None

			# Store the transition in memory.
			memory.push(state, action, next_state, reward)

			# Move to the next state.
			state = next_state

			# Perform one step of the optimization (on the policy network).
			optimize_model()
			if done:
				episode_durations.append(t + 1)
				plot_durations()
				break
		# Update the target network, copying all weights and biases in DQN.
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())

	print("Complete")
	env.render()
	env.close()
	plt.ioff()
	plt.show()

# REF [site] >> https://keras.io/examples/rl/deep_q_network_breakout/
def dqn_atari_breakout_test():
	import time, datetime
	import numpy as np
	import torch
	#from baselines.common.atari_wrappers import make_atari, wrap_deepmind
	#from stable_baselines.common.cmd_util import make_atari_env  # NOTE [info] >> TensorFlow is required.
	from stable_baselines3.common.env_util import make_atari_env

	# Install:
	#	pip install gymnasium[atari]
	#	pip install gymnasium[accept-rom-license]
	#	pip install stable-baselines3

	#-----
	# Setup

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# Number of input consecutive frames
	num_consecutive_frames = 4
	seed = 42

	# REF [site] >> https://www.gymlibrary.dev/environments/atari/breakout/
	if False:
		# Use the Baseline Atari environment because of DeepMind helper functions
		env = make_atari("BreakoutNoFrameskip-v4")
		# Warp the frames, grey scale, stake four frame and scale to smaller ratio
		env = wrap_deepmind(env, frame_stack=True, scale=True)
	else:
		env = make_atari_env("BreakoutNoFrameskip-v4")  # stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv
		#env = make_atari_env("BreakoutNoFrameskip-v4", env_kwargs=dict(render_mode="human"))
		#env = make_atari_env("BreakoutNoFrameskip-v4", env_kwargs=dict(render_mode="human", max_episode_steps=500))
	env.seed(seed)

	print(f"Observation space: {env.observation_space}.")
	print(f"Action space: {env.action_space}.")
	print(f"Action space meanings: {env.envs[0].get_action_meanings()}.")

	num_actions = env.action_space.n

	#-----
	# Implement the Deep Q-Network (DQN)

	class DQN(torch.nn.Module):
		def __init__(self, num_actions=4, num_consecutive_frames=4):
			super().__init__()

			"""
			self.conv1 = torch.nn.Conv2d(num_consecutive_frames, 32, kernel_size=8, stride=4, bias=False)
			#self.bn1 = torch.nn.BatchNorm2d(32)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
			#self.bn2 = torch.nn.BatchNorm2d(64)
			self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
			#self.bn3 = torch.nn.BatchNorm2d(64)

			self.fc1 = torch.nn.Linear(7 * 7 * 64, 512)
			self.fc2 = torch.nn.Linear(512, num_actions)
			"""
			self.conv1 = torch.nn.Conv2d(num_consecutive_frames, 32, kernel_size=8, stride=4, bias=False)
			#self.bn1 = torch.nn.BatchNorm2d(32)
			self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
			#self.bn2 = torch.nn.BatchNorm2d(64)
			self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
			#self.bn3 = torch.nn.BatchNorm2d(64)
			self.conv4 = torch.nn.Conv2d(64, 1024, kernel_size=7, stride=1, bias=False)
			#self.bn4 = torch.nn.BatchNorm2d(1024)

			self.fc = torch.nn.Linear(1024, num_actions)

		def forward(self, x):
			# [B, 4, 84, 84]: 4 consecutive frames
			"""
			#x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
			#x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
			#x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
			x = torch.nn.functional.relu(self.conv1(x))
			x = torch.nn.functional.relu(self.conv2(x))
			x = torch.nn.functional.relu(self.conv3(x))
			x = self.fc1(x.view(x.size(0), -1))
			return self.fc2(x)  # Expected values, but not actions. Actions can be chosen from the expected values.
			"""
			#x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
			#x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
			#x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
			#x = torch.nn.functional.relu(self.bn3(self.conv4(x)))
			x = torch.nn.functional.relu(self.conv1(x))
			x = torch.nn.functional.relu(self.conv2(x))
			x = torch.nn.functional.relu(self.conv3(x))
			x = torch.nn.functional.relu(self.conv4(x))
			return self.fc(x.view(x.size(0), -1))  # Expected values, but not actions. Actions can be chosen from the expected values.

	# The first model makes the predictions for Q-values which are used to make a action.
	model = DQN(num_actions, num_consecutive_frames)
	# Build a target model for the prediction of future rewards.
	# The weights of a target model get updated every 10000 steps thus when the loss between the Q-values is calculated the target Q-value is stable.
	model_target = DQN(num_actions, num_consecutive_frames)
	model.to(device)
	model_target.to(device)
	model_target.eval()

	#-----
	if True:
		# Train

		timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
		batch_size = 32  # Size of batch taken from replay buffer

		# Configuration paramaters for the whole setup
		gamma = 0.99  # Discount factor for past rewards

		epsilon = 1.0  # Epsilon greedy parameter
		epsilon_min = 0.001  # Minimum epsilon greedy parameter
		epsilon_max = 1.0  # Maximum epsilon greedy parameter
		epsilon_greedy_frames = 1000000.0  # Number of frames for exploration
		#epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
		delta_epsilon = (epsilon_max - epsilon_min) / epsilon_greedy_frames  # Amount to reduce chance of random action being taken
		epsilon_random_frames = 50000  # Number of frames to take random action and observe output

		max_steps_per_episode = 10000
		update_after_actions = 4  # Train the model after 4 actions (how often to update policy)
		update_target_network = 10000  # How often to update the target network

		# Note: The DeepMind paper suggests 1000000 however this causes memory issues
		max_memory_length = 1000000  # Maximum replay length

		if True:
			# Initialize weights
			for param in model.parameters():
				if param.dim() > 1 and param.requires_grad:
					torch.nn.init.xavier_normal_(param, gain=2.0)
			print("Model weights initialized.")

		# Improves training time
		# In the DeepMind paper they use RMSProp however then Adam optimizer
		#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		#max_clipping_norm = None
		max_clipping_norm = 1.0

		# Using Huber loss for stability
		loss_function = torch.nn.HuberLoss()

		# Experience replay buffers
		state_history = []
		action_history = []
		reward_history = []
		state_next_history = []
		done_history = []

		if False:
			# Initialize experience replay buffer
			print("Initializing experience replay buffer...")
			start_time = time.time()
			while True:
				state = env.reset()  # [#envs, H, W, 1]
				state = state.repeat(repeats=num_consecutive_frames, axis=-1)  # N consecutive frames, [#envs, H, W, #frames]

				for _ in range(max_steps_per_episode):
					# Take random action
					action = [np.random.choice(num_actions)]

					# Apply the sampled action in our environment
					state_next, reward, done, info = env.step(action)
					#state_next = np.concatenate([state[...,1:], state_next], axis=-1)  # N consecutive frames
					state_next = np.concatenate([state_next, state[...,:-1]], axis=-1)  # N consecutive frames

					if reward[0] > 0:
						# Save actions and states in replay buffer
						state_history.append(state)
						action_history.append(action)
						reward_history.append(reward)
						state_next_history.append(state_next)
						done_history.append(done)

					if done:
						break

					state = state_next

				if len(state_history) >= max_memory_length // 2:
				#if len(state_history) >= max_memory_length // 10:
					break
			print(f"Experience replay buffer initialized: {time.time() - start_time} secs.")
			print(f"Experience replay buffer: initial size = {len(state_history)}, max size = {max_memory_length}.")

		episode_reward_history = []
		running_reward = 0
		episode_count = 0
		frame_count = 0

		print("Training...")
		start_time = time.time()
		while True:  # Run until solved
			state = env.reset()  # [#envs, H, W, 1]
			state = state.repeat(repeats=num_consecutive_frames, axis=-1)  # N consecutive frames, [#envs, H, W, #frames]

			episode_reward = 0
			for episode_step in range(1, max_steps_per_episode + 1):
				#screen = env.render()  # Adding this line would show the attempts of the agent in a pop up window

				frame_count += 1

				# Use epsilon-greedy for exploration
				if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
					# Take random action
					action = [np.random.choice(num_actions)]
					#action = [env.action_space.sample()]
				else:
					# Predict action Q-values
					# From environment state
					state_tensor = torch.tensor(state.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
					model.eval()
					with torch.no_grad():
						action_probs = model(state_tensor)
					# Take best action
					action = [torch.argmax(action_probs[0], dim=-1).cpu().numpy()]

				# Decay probability of taking random action
				#epsilon = max(epsilon - epsilon_interval / epsilon_greedy_frames, epsilon_min)
				epsilon = max(epsilon - delta_epsilon, epsilon_min)
				#if frame_count % 500 == 0: epsilon = max(epsilon * 0.999, epsilon_min)

				# Apply the sampled action in our environment
				state_next, reward, done, info = env.step(action)
				#state_next = np.concatenate([state[...,1:], state_next], axis=-1)  # N consecutive frames
				state_next = np.concatenate([state_next, state[...,:-1]], axis=-1)  # N consecutive frames

				episode_reward += reward

				# Save actions and states in replay buffer
				state_history.append(state)
				action_history.append(action)
				reward_history.append(reward)
				state_next_history.append(state_next)
				done_history.append(done)
				state = state_next

				# Update every fourth frame and once batch size is over 32
				if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
					# Get indices of samples for replay buffers
					indices = np.random.choice(range(len(done_history)), size=batch_size)

					# Using list comprehension to sample from replay buffer
					state_sample = np.concatenate([state_history[i] for i in indices], axis=0)  # [B, H, W, #frames]
					action_sample = np.concatenate([action_history[i] for i in indices], axis=0)  # [B]
					reward_sample = np.concatenate([reward_history[i] for i in indices], axis=0)  # [B]
					state_next_sample = np.concatenate([state_next_history[i] for i in indices], axis=0)  # [B, H, W, #frames]
					done_sample = np.concatenate([done_history[i] for i in indices], axis=0, dtype=np.float32)  # [B]

					# Build the updated Q-values for the sampled future states
					# Use the target model for stability
					state_next_sample = torch.tensor(state_next_sample.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
					with torch.no_grad():
						future_rewards = model_target(state_next_sample)
					# Q value = reward + discount factor * expected future reward
					updated_q_values = torch.tensor(reward_sample) + gamma * torch.max(future_rewards.cpu(), dim=-1).values

					# If final frame set the last value to -1
					updated_q_values = updated_q_values * (1 - done_sample) - done_sample

					# Create a mask so we only calculate loss on the updated Q-values
					masks = torch.nn.functional.one_hot(torch.tensor(action_sample), num_classes=num_actions)

					model.train()

					# Zero the parameter gradients
					optimizer.zero_grad()

					# Forward + backward + optimize
					# Train the model on the states and updated Q-values
					state_sample = torch.tensor(state_sample.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
					q_values = model(state_sample)
					# Apply the masks to the Q-values to get the Q-value for action taken
					q_action = torch.sum(torch.mul(q_values.cpu(), masks), dim=-1)

					loss = loss_function(updated_q_values, q_action)
					loss.backward()
					if max_clipping_norm is not None:
						torch.nn.utils.clip_grad_norm_(model.parameters(), max_clipping_norm)
					optimizer.step()

				if frame_count % update_target_network == 0:
					# Update the the target network with new weights
					model_target.load_state_dict(model.state_dict())

					# Log details
					print(f"Running reward = {running_reward:.4f} at episode {episode_count}, frame {frame_count}: epsilon = {epsilon:.4f}.")

				# Limit the state and reward history
				if len(reward_history) > max_memory_length:
					del state_history[:1]
					del action_history[:1]
					del reward_history[:1]
					del state_next_history[:1]
					del done_history[:1]

				if done:
					"""
					Examples:
						Done: [ True], Reward: [0.], Info: [{'lives': 4, 'episode_frame_number': 415, 'frame_number': 1843626, 'TimeLimit.truncated', False}], Episode reward: [2.].
						Done: [ True], Reward: [0.], Info: [{'lives': 3, 'episode_frame_number': 519, 'frame_number': 1843730, 'TimeLimit.truncated', False}], Episode reward: [0.].
						Done: [ True], Reward: [0.], Info: [{'lives': 2, 'episode_frame_number': 1211, 'frame_number': 1844422, 'TimeLimit.truncated', False}], Episode reward: [3.].
						Done: [ True], Reward: [0.], Info: [{'lives': 1, 'episode_frame_number': 1315, 'frame_number': 1844526, 'TimeLimit.truncated', False}], Episode reward: [0.].
						Done: [ True], Reward: [0.], Info: [{'lives': 0, 'episode_frame_number': 1409, 'frame_number': 1844620, 'episode': {'r': 5.0, 'l': 1409, 't': 2022.855433}, 'TimeLimit.truncated': False}], Episode reward: [0.].

						Done: [ True], Reward: [0.], Info: [{'lives': 4, 'episode_frame_number': 300, 'frame_number': 2137036, 'TimeLimit.truncated': False}], Episode reward: [1.].
						Done: [ True], Reward: [0.], Info: [{'lives': 3, 'episode_frame_number': 828, 'frame_number': 2137564, 'TimeLimit.truncated': False}], Episode reward: [3.].
						Done: [ True], Reward: [0.], Info: [{'lives': 2, 'episode_frame_number': 932, 'frame_number': 2137668, 'TimeLimit.truncated': False}], Episode reward: [0.].
						Done: [ True], Reward: [0.], Info: [{'lives': 1, 'episode_frame_number': 1352, 'frame_number': 2138088, 'TimeLimit.truncated': False}], Episode reward: [2.].
						Done: [ True], Reward: [0.], Info: [{'lives': 0, 'episode_frame_number': 1446, 'frame_number': 2138182, 'episode': {'r': 6.0, 'l': 1446, 't': 2390.495793}, 'TimeLimit.truncated': False}], Episode reward: [0.].
					"""

					#del info[0]["terminal_observation"]
					#print(f"Done: {done}, Reward: {reward}, Info: {info}, Episode reward: {episode_reward}.")
					break

			# Update running reward to check condition for solving
			episode_reward_history.append(episode_reward)
			if len(episode_reward_history) > 100:
				del episode_reward_history[:1]
			running_reward = np.mean(episode_reward_history)

			episode_count += 1

			# Max average reward = 108 / 5 = 21.6.
			#	#bricks = 18 x 6 = 108, #lives = 5.
			#	Random actions should be considered.
			# FIXME [restore] >>
			#if running_reward > 40:  # Condition to consider the task solved
			#if running_reward > 4:  # Condition to consider the task solved. When epsilon_min = 0.1
			#if running_reward > 6:  # Condition to consider the task solved. When epsilon_min = 0.01
			if running_reward > 8:  # Condition to consider the task solved. When epsilon_min = 0.001
				print(f"Solved at episode {episode_count}!")
				break
		print(f"Trained: {time.time() - start_time:} secs.")
		env.close()

		if True:
			# Save the weights
			torch.save({"state_dict": model.state_dict()}, f"./dqn_breakout_{timestamp}.pth")
			torch.save({"state_dict": model_target.state_dict()}, f"./dqn_breakout_target_{timestamp}.pth")

	#-----
	if False:
		# Load the weights
		loaded_data = torch.load("./dqn_breakout.pth", map_location=device)
		model.load_state_dict(loaded_data['state_dict'])
		loaded_data = torch.load("./dqn_breakout_target.pth", map_location=device)
		model_target.load_state_dict(loaded_data['state_dict'])

		model.to(device)
		model_target.to(device)

	model.eval()
	model_target.eval()

	env = make_atari_env("BreakoutNoFrameskip-v4", env_kwargs=dict(render_mode="human"))
	state = env.reset()  # [#envs, H, W, 1]
	state = state.repeat(repeats=num_consecutive_frames, axis=-1)  # N consecutive frames, [#envs, H, W, #frames]
	episode_reward = 0
	while True:
		#env.render()

		state_tensor = torch.tensor(state.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
		if True:
			with torch.no_grad():
				values = model(state_tensor)
		else:
			with torch.no_grad():
				values = model(state_tensor)
				target_values = model_target(state_tensor)
			print(f"Target value: {target_values.cpu().numpy()}, Value: {values.cpu().numpy()}.")
		action = [torch.argmax(values[0], dim=-1).cpu().numpy()]

		state_next, reward, done, info = env.step(action)
		episode_reward += reward

		if done:
			#del info[0]["terminal_observation"]
			print(f"Done: {done}, Reward: {reward}, Info: {info}, Episode reward: {episode_reward}.")
			state = env.reset()  # [#envs, H, W, 1]
			state = state.repeat(repeats=num_consecutive_frames, axis=-1)  # N consecutive frames, [#envs, H, W, #frames]
			episode_reward = 0
		else:
			#state = np.concatenate([state[...,1:], state_next], axis=-1)  # N consecutive frames
			state = np.concatenate([state_next, state[...,:-1]], axis=-1)  # N consecutive frames

	#env.render()
	env.close()

# REF [site] >> https://keras.io/examples/rl/ddpg_pendulum/
def ddpg_inverted_pendulum_test():
	import time, datetime
	import numpy as np
	import torch
	import gym
	#import gymnasium as gym
	import matplotlib.pyplot as plt

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

	# REF [site] >> https://www.gymlibrary.dev/environments/classic_control/pendulum/
	if False:
		# For gym
		gym.envs.register(
			id="Pendulum-v1",
			entry_point="gym.envs.classic_control:PendulumEnv",
			max_episode_steps=300,  # Default: 200
			#reward_threshold=-110.0,  # Default: None
		)
	env = gym.make("Pendulum-v1")
	#env = gym.make("Pendulum-v1", render_mode="human")
	#env = gym.make("Pendulum-v1", render_mode="human", g=10.0, max_episode_steps=500)

	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]
	lower_bound = env.action_space.low[0]
	upper_bound = env.action_space.high[0]

	print(f"Observation space: {env.observation_space}.")
	print(f"Action space: {env.action_space}.")
	print(f"Dimension of state space = {num_states}.")
	print(f"Dimension of action space = {num_actions}.")
	print(f"Min & max value of actions = [{lower_bound}, {upper_bound}].")
	print(f"Render mode = {env.render_mode}.")
	print(f"Max episode steps = {env.spec.max_episode_steps}.")
	print(f"Reward threshold = {env.spec.reward_threshold}.")

	# An Ornstein-Uhlenbeck process for generating noise
	class OUActionNoise:
		def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
			self.theta = theta
			self.mean = mean
			self.std_dev = std_deviation
			self.dt = dt
			self.x_initial = x_initial
			self.reset()

		def __call__(self):
			# Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
			x = (
				self.x_prev
				+ self.theta * (self.mean - self.x_prev) * self.dt
				+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
			)
			# Store x into x_prev
			# Makes next noise dependent on current one
			self.x_prev = x
			return x

		def reset(self):
			if self.x_initial is not None:
				self.x_prev = self.x_initial
			else:
				self.x_prev = np.zeros_like(self.mean)

	# Experience replay
	class Buffer:
		def __init__(self, buffer_capacity=100000, batch_size=64):
			# Number of "experiences" to store at max
			self.buffer_capacity = buffer_capacity
			# Num of tuples to train on.
			self.batch_size = batch_size

			# Its tells us num of times record() was called
			self.buffer_counter = 0

			# Instead of list of tuples as the exp.replay concept go
			# We use different np.arrays for each tuple element
			self.state_buffer = np.zeros((self.buffer_capacity, num_states))
			self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
			self.reward_buffer = np.zeros((self.buffer_capacity, 1))
			self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

		# Takes (s, a, r, s') obervation tuple as input
		def record(self, obs_tuple):
			# Set index to zero if buffer_capacity is exceeded, replacing old records
			index = self.buffer_counter % self.buffer_capacity

			self.state_buffer[index] = obs_tuple[0]
			self.action_buffer[index] = obs_tuple[1]
			self.reward_buffer[index] = obs_tuple[2]
			self.next_state_buffer[index] = obs_tuple[3]

			self.buffer_counter += 1

		def update(self, state_batch, action_batch, reward_batch, next_state_batch):
			# Training and updating Actor & Critic networks.
			# See Pseudo Code in the paper.

			state_batch = state_batch.to(device)
			action_batch = action_batch.to(device)
			reward_batch = reward_batch.to(device)
			next_state_batch = next_state_batch.to(device)

			critic_model.train()
			actor_model.train()

			# Zero the parameter gradients
			critic_optimizer.zero_grad()

			# Forward + backward + optimize
			with torch.no_grad():
				target_actions = target_actor(next_state_batch)
				y = reward_batch + gamma * target_critic(next_state_batch, target_actions)
			critic_value = critic_model(state_batch, action_batch)

			critic_loss = critic_loss_function(y, critic_value)
			#critic_loss = torch.mean(torch.square(y - critic_value))
			critic_loss.backward()
			critic_optimizer.step()

			# Zero the parameter gradients
			actor_optimizer.zero_grad()

			# Forward + backward + optimize
			actions = actor_model(state_batch)
			critic_value = critic_model(state_batch, actions)

			# Used `-value` as we want to maximize the value given by the critic for our actions
			actor_loss = actor_loss_function(critic_value)
			#actor_loss = -torch.mean(critic_value)
			actor_loss.backward()
			actor_optimizer.step()

		# We compute the loss and update parameters
		def learn(self):
			# Get sampling range
			record_range = min(self.buffer_counter, self.buffer_capacity)
			# Randomly sample indices
			batch_indices = np.random.choice(record_range, self.batch_size)

			# Convert to tensors
			state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
			action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
			reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
			next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)

			self.update(state_batch, action_batch, reward_batch, next_state_batch)

	# This update target parameters slowly
	# Based on rate `tau`, which is much less than one.
	def update_target(target_weights, weights, tau):
		for (a, b) in zip(target_weights, weights):
			a.copy_(b * tau + a * (1 - tau))

	# Actor and critic networks
	class Actor(torch.nn.Module):
		def __init__(self, num_states, upper_bound):
			super().__init__()

			self.upper_bound = upper_bound

			self.fc1 = torch.nn.Linear(num_states, 256)
			self.fc2 = torch.nn.Linear(256, 256)
			self.fc3 = torch.nn.Linear(256, 1)

			# Initialize weights between -3e-3 and 3-e3
			for param in self.fc3.parameters():
				if param.dim() > 1 and param.requires_grad:
					torch.nn.init.uniform_(param, a=-0.003, b=0.003)

		def forward(self, x):
			x = torch.nn.functional.relu(self.fc1(x))
			x = torch.nn.functional.relu(self.fc2(x))
			x = torch.tanh(self.fc3(x))
			# Our upper bound is 2.0 for Pendulum
			return x * self.upper_bound

	class Critic(torch.nn.Module):
		def __init__(self, num_states, num_actions):
			super().__init__()

			self.state_fc1 = torch.nn.Linear(num_states, 16)
			self.state_fc2 = torch.nn.Linear(16, 32)

			self.action_fc = torch.nn.Linear(num_actions, 32)

			self.fc1 = torch.nn.Linear(64, 256)
			self.fc2 = torch.nn.Linear(256, 256)
			self.fc3 = torch.nn.Linear(256, 1)

		def forward(self, state, action):
			state = torch.nn.functional.relu(self.state_fc1(state))
			state = torch.nn.functional.relu(self.state_fc2(state))

			action = torch.nn.functional.relu(self.action_fc(action))

			x = torch.cat([state, action], dim=-1)
			x = torch.nn.functional.relu(self.fc1(x))
			x = torch.nn.functional.relu(self.fc2(x))
			return self.fc3(x)

	# Sample an action from our actor network plus some noise for exploration
	def policy(state, noise_object):
		actor_model.eval()
		with torch.no_grad():
			sampled_actions = actor_model(state.to(device)).squeeze()
		noise = noise_object()
		# Adding noise to action
		sampled_actions = sampled_actions.cpu().numpy() + noise

		# We make sure action is within bounds
		legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

		return [np.squeeze(legal_action)]

	# Training hyperparameters
	std_dev = 0.2
	ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

	actor_model = Actor(num_states, upper_bound)
	critic_model = Critic(num_states, num_actions)
	target_actor = Actor(num_states, upper_bound)
	target_critic = Critic(num_states, num_actions)
	actor_model.to(device)
	critic_model.to(device)
	target_actor.to(device)
	target_critic.to(device)
	target_actor.eval()
	target_critic.eval()

	# Making the weights equal initially
	target_actor.load_state_dict(actor_model.state_dict())
	target_critic.load_state_dict(critic_model.state_dict())

	# Learning rate for actor-critic models
	actor_lr = 0.001
	critic_lr = 0.002

	actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=actor_lr)
	critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)

	def negative_mean_loss(input):
		return -torch.mean(input)

	actor_loss_function = negative_mean_loss
	critic_loss_function = torch.nn.MSELoss(reduction="mean")

	total_episodes = 100
	# Discount factor for future rewards
	gamma = 0.99
	# Used to update target networks
	tau = 0.005

	buffer = Buffer(buffer_capacity=50000, batch_size=64)

	#-----
	# Main training loop

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []

	# Takes about 4 min to train
	print("Training...")
	start_time = time.time()
	for ep in range(total_episodes):
		#prev_state = env.reset()
		prev_state, info = env.reset()  # For gym & gymnasium
		episodic_reward = 0

		for episode_step in range(1, env.spec.max_episode_steps + 1):
			# Uncomment this to see the Actor in action
			# But not in a Python notebook.
			#env.render()  # Slow.

			action = policy(torch.tensor(prev_state, dtype=torch.float32).unsqueeze(dim=0), ou_noise)

			# Recieve state and reward from environment
			#state, reward, done, info = env.step(action)
			state, reward, terminated, truncated, info = env.step(action)  # For gym & gymnasium

			#if episode_step >= env.spec.max_episode_steps and not terminated and not truncated:
			#	#done = True
			#	truncated = True
			#	info.update({"TimeLimit.truncated_by_user": True})

			buffer.record((prev_state, action, reward, state))
			episodic_reward += reward

			buffer.learn()
			with torch.no_grad():
				update_target(target_actor.parameters(), actor_model.parameters(), tau)
				update_target(target_critic.parameters(), critic_model.parameters(), tau)

			# End this episode when `done` is True
			#if done:
			if terminated or truncated:
				#print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}, Info: {info}.")
				break

			prev_state = state

		ep_reward_list.append(episodic_reward)

		# Mean of last 40 episodes
		avg_reward = np.mean(ep_reward_list[-40:])
		print(f"Episode {ep}: avg reward = {avg_reward}.")
		avg_reward_list.append(avg_reward)
	print(f"Trained: {time.time() - start_time} secs.")
	env.close()

	if True:
		# Plotting graph
		# Episodes versus Avg. Rewards
		plt.plot(avg_reward_list)
		plt.xlabel("Episode")
		plt.ylabel("Avg. Epsiodic Reward")
		plt.show()

	if True:
		# Save the weights
		torch.save({"state_dict": actor_model.state_dict()}, f"./ddpg_inverted_pendulum_actor_{timestamp}.pth")
		torch.save({"state_dict": critic_model.state_dict()}, f"./ddpg_inverted_pendulum_critic_{timestamp}.pth")

		torch.save({"state_dict": target_actor.state_dict()}, f"./ddpg_inverted_pendulum_target_actor_{timestamp}.pth")
		torch.save({"state_dict": target_critic.state_dict()}, f"./ddpg_inverted_pendulum_target_critic_{timestamp}.pth")

	#-----
	if False:
		# Load the weights
		loaded_data = torch.load("./ddpg_inverted_pendulum_actor_.pth", map_location=device)
		actor_model.load_state_dict(loaded_data['state_dict'])
		loaded_data = torch.load("./ddpg_inverted_pendulum_critic_.pth", map_location=device)
		critic_model.load_state_dict(loaded_data['state_dict'])

		loaded_data = torch.load("./ddpg_inverted_pendulum_target_actor_.pth", map_location=device)
		target_actor.load_state_dict(loaded_data['state_dict'])
		loaded_data = torch.load("./ddpg_inverted_pendulum_target_critic_.pth", map_location=device)
		target_critic.load_state_dict(loaded_data['state_dict'])

		actor_model.to(device)
		critic_model.to(device)
		target_actor.to(device)
		target_critic.to(device)

	actor_model.eval()
	critic_model.eval()
	target_actor.eval()
	target_critic.eval()

	env = gym.make("Pendulum-v1", render_mode="human", g=10.0, max_episode_steps=500)
	obs, info = env.reset()
	while True:
		#env.render()

		states = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=device)
		if True:
			with torch.no_grad():
				actions = actor_model(states)
		else:
			with torch.no_grad():
				actions = actor_model(states)
				values = critic_model(states, actions)
				target_actions = target_actor(states)
				target_values = target_critic(states, target_actions)
			print(f"Target action: {target_actions.cpu().numpy()}, Action: {actions.cpu().numpy()}, Target value: {target_values.cpu().numpy()}, Value: {values.cpu().numpy()}.")
		action = actions.squeeze(dim=0).cpu().numpy()

		obs, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			print(f"Terminated: {terminated}, Truncated: {truncated}, Reward: {reward}, Info: {info}.")
			obs, info = env.reset()

	#env.render()
	env.close()

def main():
	# Value-function-based algorithm

	# Deep Q-Network (DQN)
	#dqn_cart_pole_tutorial()  # More structured implementation.
	#dqn_atari_breakout_test()  # Naive low-level implementation.

	#-----
	# Policy gradient algorithm

	# REINFORCE
	#	Training using REINFORCE for Mujoco
	#		https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

	# Advantage actor-critic (A2C)
	#	Training A2C with Vector Envs and Domain Randomization
	#		https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/

	# Deep deterministic policy gradient (DDPG) algorithm
	#	Model-free, off-policy actor-critic algorithm
	ddpg_inverted_pendulum_test()

	# Twin delayed deep deterministic (TD3) policy gradient algorithm
	#	Twin delayed DDPG
	#	TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
