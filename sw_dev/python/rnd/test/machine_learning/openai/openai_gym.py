#!/usr/bin/env python

import gym

# REF [site] >> https://gym.openai.com/docs/
def env_info():
	space = gym.spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}.
	x = space.sample()
	assert space.contains(x)
	assert space.n == 8

	print(gym.envs.registry.all())

	#--------------------
	env = gym.make('CartPole-v1')

	print('Action space =', env.action_space)
	print('Observation space =', env.observation_space)

	print('Observation bounds (high) =', env.observation_space.high)
	print('Observation bounds (low) =', env.observation_space.low)

# REF [site] >> https://gym.openai.com/docs/
def simple_agent_environment_loop():
	env = gym.make('CartPole-v1')
	env.reset()

	for i_episode in range(20):
		observation = env.reset()  # Return an initial observation.
		for t in range(100):
			env.render()
			print('Observation =', observation)
			action = env.action_space.sample()  # Take a random action.
			observation, reward, done, info = env.step(action)
			if done:
				print('Episode finished after {} timesteps'.format(t + 1))
				break

def main():
	env_info()
	#simple_agent_environment_loop()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
