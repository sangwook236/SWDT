#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gym

# REF [site] >> https://gym.openai.com/docs/
def env_info():
	space = gym.spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}.
	x = space.sample()
	assert space.contains(x)
	assert space.n == 8

	print(gym.envs.registry)  # dict.

	#--------------------
	if True:
		# Discrete action, continuous observation.

		env = gym.make('CartPole-v1')
		#env = gym.make('CartPole-v1', render_mode='human')
		#env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=500)

		assert env.action_space.n == 2

		print(f'Action space: {env.action_space}.')
		print(f'Dimension of action space = {env.action_space.shape}.')
		#print(f'Action bounds: (low, high) = ({env.action_space.low}, {env.action_space.high}).')  # AttributeError: 'Discrete' object has no attribute 'low' & 'high'.
		#print(f'Action space meanings: {env.get_action_meanings()}.')  # AttributeError: 'CartPoleEnv' object has no attribute 'get_action_meanings'.

		print(f'Observation space: {env.observation_space}.')
		print(f'Dimension of observation space = {env.observation_space.shape}.')
		print(f'Observation bounds: (low, high) = ({env.observation_space.low}, {env.observation_space.high}).')

		print(f'Metadata: {env.metadata}.')
		print(f'Render mode = {env.render_mode}.')

		print(f'Spec: {env.spec}.')
		print(f'Max episode steps = {env.spec.max_episode_steps}.')
		print(f'Reward threshold = {env.spec.reward_threshold}.')

	if True:
		# Continuous action, continuous observation.

		env = gym.make('Pendulum-v1')
		#env = gym.make('Pendulum-v1', render_mode='human')
		#env = gym.make('Pendulum-v1', render_mode='human', g=10.0, max_episode_steps=500)

		print(f'Action space: {env.action_space}.')
		print(f'Dimension of action space = {env.action_space.shape}.')
		print(f'Action bounds: (low, high) = ({env.action_space.low}, {env.action_space.high}.')
		#print(f'Action space meanings: {env.get_action_meanings()}.')  # AttributeError: 'PendulumEnv' object has no attribute 'get_action_meanings'.

		print(f'Observation space: {env.observation_space}.')
		print(f'Dimension of observation space = {env.observation_space.shape}.')
		print(f'Observation bounds: (low, high) = ({env.observation_space.low}, {env.observation_space.high}.')

		print(f'Metadata: {env.metadata}.')
		print(f'Render mode = {env.render_mode}.')

		print(f'Spec: {env.spec}.')
		print(f'Max episode steps = {env.spec.max_episode_steps}.')
		print(f'Reward threshold = {env.spec.reward_threshold}.')

	if True:
		# Discrete action, discrete(image) observation.

		env = gym.make('Breakout-v4')
		#env = gym.make('Breakout-v4', render_mode='human')
		#env = gym.make('Breakout-v4', render_mode='human', max_episode_steps=500)

		assert env.action_space.n == 4

		print(f'Action space: {env.action_space}.')
		print(f'Dimension of action space = {env.action_space.shape}.')
		#print(f'Action bounds: (low, high) = ({env.action_space.low}, {env.action_space.high}.')  # AttributeError: 'Discrete' object has no attribute 'low' & 'high'.
		print(f'Action space meanings: {env.get_action_meanings()}.')

		print(f'Observation space: {env.observation_space}.')
		print(f'Dimension of observation space = {env.observation_space.shape}.')
		print(f'Observation bounds: (low, high) = ({env.observation_space.low}, {env.observation_space.high}.')

		print(f'Metadata: {env.metadata}.')
		print(f'Render mode = {env.render_mode}.')

		print(f'Spec: {env.spec}.')
		print(f'Max episode steps = {env.spec.max_episode_steps}.')
		print(f'Reward threshold = {env.spec.reward_threshold}.')

# REF [site] >> https://gym.openai.com/docs/
def simple_agent_environment_loop():
	env = gym.make('CartPole-v1')
	#env.reset()

	for episode_step in range(20):
		observation = env.reset()  # Return an initial observation.
		for step in range(100):
			env.render()
			print(f'Observation = {observation}.')
			action = env.action_space.sample()  # Take a random action.
			observation, reward, done, info = env.step(action)
			if done:
				print(f'Episode finished after {step + 1} timesteps.')
				break

def main():
	env_info()
	#simple_agent_environment_loop()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
