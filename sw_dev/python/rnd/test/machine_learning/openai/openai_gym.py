#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gym
#import gymnasium as gym

# REF [site] >> https://gymnasium.farama.org/content/basic_usage/
def basic_usage_1():
	# Initializing environments.
	# REF [site] >> https://gymnasium.farama.org/environments/classic_control/cart_pole/
	env = gym.make('CartPole-v1')

	# Interacting with the environment.
	# REF [site] >> https://gymnasium.farama.org/environments/box2d/lunar_lander/
	env = gym.make('LunarLander-v2', render_mode='human')
	observation, info = env.reset()

	for _ in range(1000):
		action = env.action_space.sample()  # Agent policy that uses the observation and info.
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()

	env.close()

	# Modifying the environment.
	from gymnasium.wrappers import FlattenObservation

	env = gym.make('CarRacing-v2')
	print(f'{env.observation_space.shape=}.')

	wrapped_env = FlattenObservation(env)
	print(f'{wrapped_env.observation_space.shape=}.')

	print(f'{wrapped_env=}.')
	print(f'{wrapped_env.unwrapped=}.')

# REF [site] >> https://www.gymlibrary.dev/content/basic_usage/
def basic_usage_2():
	# Initializing environments.

	# REF [site] >> https://gymnasium.farama.org/environments/classic_control/cart_pole/
	env = gym.make('CartPole-v0')

	#-----
	# Interacting with the environment.

	# REF [site] >> https://gymnasium.farama.org/environments/box2d/lunar_lander/
	env = gym.make('LunarLander-v2', render_mode='human')
	env.action_space.seed(42)

	observation, info = env.reset(seed=42)

	for _ in range(1000):
		observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

		if terminated or truncated:
			observation, info = env.reset()

	env.close()

	#-----
	# Checking API-conformity.

	from gym.utils.env_checker import check_env
	check_env(env, warn=None, skip_render_check=False)

	#-----
	# Spaces.

	from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete

	observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
	print(observation_space.sample())

	observation_space = Discrete(4)
	print(observation_space.sample())

	observation_space = Discrete(5, start=-2)
	print(observation_space.sample())

	observation_space = Dict({'position': Discrete(2), 'velocity': Discrete(3)})
	print(observation_space.sample())

	observation_space = Tuple((Discrete(2), Discrete(3)))
	print(observation_space.sample())

	observation_space = MultiBinary(5)
	print(observation_space.sample())

	observation_space = MultiDiscrete([5, 2, 2])
	print(observation_space.sample())

	#-----
	# Wrappers.
	#	Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly.
	#	Using wrappers will allow you to avoid a lot of boilerplate code and make your environment more modular.
	#	Wrappers can also be chained to combine their effects.
	#	Most environments that are generated via gym.make will already be wrapped by default.

	from gym.wrappers import RescaleAction

	base_env = gym.make('BipedalWalker-v3')
	print(f'{base_env.action_space=}.')

	wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
	print(f'{wrapped_env.action_space=}.')

	print(f'{wrapped_env=}.')
	print(f'{wrapped_env.unwrapped=}.')

	#-----
	# Playing within an environment.

	# You can also play the environment using your keyboard using the play function in gym.utils.play.

	from gym.utils.play import play
	import pygame

	play(gym.make('Pong-v4', render_mode='human'))

	mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
	play(gym.make('CartPole-v0', render_mode='human'), keys_to_action=mapping)

	# If you wish to plot real time statistics as you play, you can use gym.utils.play.PlayPlot.
	# Here's some sample code for plotting the reward for last 5 second of gameplay.

	from gym.utils.play import PlayPlot

	def callback(obs_t, obs_tp1, action, rew, done, info):
		return [rew,]

	plotter = PlayPlot(callback, 30 * 5, ['reward'])
	env = gym.make('Pong-v4')
	play(env, callback=plotter.callback)

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

		# REF [site] >> https://gymnasium.farama.org/environments/classic_control/cart_pole/
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

		# REF [site] >> https://gymnasium.farama.org/environments/classic_control/pendulum/
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

		# REF [site] >> https://gymnasium.farama.org/environments/atari/breakout/
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
	# REF [site] >> https://gymnasium.farama.org/environments/classic_control/cart_pole/
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
	# REF [site] >>
	#	https://gymnasium.farama.org/
	#	https://www.gymlibrary.dev/

	basic_usage_1()
	basic_usage_2()

	#-----
	# Environments:
	#	Classic Control:
	#		https://gymnasium.farama.org/environments/classic_control/
	#		Acrobot, Cart Pole, Mountain Car Continuous, Mountain Car, Pendulum.
	#	Box2D:
	#		https://gymnasium.farama.org/environments/box2d/
	#		Bipedal Walker, Car Racing, Lunar Lander.
	#	Toy Text:
	#		https://gymnasium.farama.org/environments/toy_text/
	#		Blackjack, Taxi, Cliff Walking, Frozen Lake.
	#	MuJoCo:
	#		https://gymnasium.farama.org/environments/mujoco/
	#		Ant, Half Cheetah, Hopper, Humanoid, Humanoid Standup, Inverted Double Pendulum, Inverted Pendulum, Pusher, Reacher, Swimmer, Walker2D.
	#	Atari:
	#		https://gymnasium.farama.org/environments/atari/
	#		Adventure, AirRaid, Alien, Amidar, Assault, Asterix, Asteroids, Atlantis, Atlantis2, Backgammon, BankHeist, BasicMath, BattleZone, BeamRider, Berzerk, Blackjack, Bowling, Boxing, Breakout,
	#		Carnival, Casino, Centipede, ChopperCommand, CrazyClimber, Crossbow, Darkchambers, Defender, DemonAttack, DonkeyKong, DoubleDunk, Earthworld, ElevatorAction, Enduro, Entombed, Et,
	# 		FFishingDerby, FlagCapture, Freeway, Frogger, Frostbite, Galaxian, Gopher, Gravitar, Hangman, HauntedHouse, Hero, HumanCannonball, IceHockey, Jamesbond, JourneyEscape,
	#		Kaboom, Kangaroo, KeystoneKapers, KingKong, Klax, Koolaid, Krull, KungFuMaster, LaserGates, LostLuggage, MarioBros, MiniatureGolf, MontezumaRevenge, MrDo, MsPacman, NameThisGame,
	#		Othello, Pacman, Phoenix, Pitfall, Pitfall2, Pong, Pooyan, PrivateEye, Qbert, Riverraid, RoadRunner, Robotank, Seaquest, SirLancelot, Skiing, Solaris, SpaceInvaders, SpaceWar, StarGunner, Superman, Surround,
	#		Tennis, Tetris, TicTacToe3D, TimePilot, Trondead, Turmoil, Tutankham, UpNDown, Venture, VideoCheckers, VideoChess, VideoCube, VideoPinball, WizardOfWor, WordZapper, YarsRevenge, Zaxxon.
	#	Third Party Environments:
	#		https://gymnasium.farama.org/environments/third_party_environments/
	#		https://www.gymlibrary.dev/environments/third_party_environments/
	#		Video Game environments:
	#			ViZDoom, MineRL, Procgen, Unity ML Agents.
	#		Robotics environments:
	#			PyFlyt, MarsExplorer, robo-gym, DexterousHands, OmniIsaacGymEnvs.
	#		Autonomous Driving environments:
	#			CommonRoad-RL, racing_dreamer, racecar_gym.
	#		Classic Environments (board, card, etc. games):
	#			RubiksCubeGym, GymGo, MindMaker Unreal Engine Plugin.
	#		Other environments:
	#			CompilerGym, DACBench, NLPGym, ShinRL, GymFC.

	env_info()
	#simple_agent_environment_loop()

	#-----
	# Gymnasium basics.
	#	https://gymnasium.farama.org/tutorials/gymnasium_basics/

	# Handling time limits.
	#	https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
	# Implementing custom wrappers.
	#	https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/
	# Make your own custom environment.
	#	https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
	# Training A2C with vector envs and domain randomization.
	#	https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/

	#-----
	# Training Agents.
	#	https://gymnasium.farama.org/tutorials/training_agents/

	# Training using REINFORCE for Mujoco.
	#	https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/
	# Solving Blackjack with Q-Learning.
	#	https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
	# Frozenlake benchmark.
	#	https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/

	#-----
	# Custom Environment.

	# Make your own custom environment:
	#	https://www.gymlibrary.dev/content/environment_creation/
	# Vectorising your environments:
	#	https://www.gymlibrary.dev/content/vectorising/

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
