#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, itertools, datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl import config
from finrl import config_tickers
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

#matplotlib.use("Agg")
#%matplotlib inline

# REF [site] >> https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/1-Introduction/Stock_NeurIPS2018.ipynb
#	Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading.
def multiple_stock_trading_tutorial():
	#sys.path.append("../FinRL")

	#--------------------
	# Create folders.

	if not os.path.exists("./" + config.DATA_SAVE_DIR):
		os.makedirs("./" + config.DATA_SAVE_DIR)
	if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
		os.makedirs("./" + config.TRAINED_MODEL_DIR)
	if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
		os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
	if not os.path.exists("./" + config.RESULTS_DIR):
		os.makedirs("./" + config.RESULTS_DIR)

	#--------------------
	# Download data.

	# From config.py, TRAIN_START_DATE is a string.
	print("config.TRAIN_START_DATE = {}.".format(config.TRAIN_START_DATE))
	# From config.py, TRAIN_END_DATE is a string.
	print("config.TRAIN_END_DATE = {}.".format(config.TRAIN_END_DATE))
	print("config_tickers.DOW_30_TICKER = {}.".format(config_tickers.DOW_30_TICKER))

	df = YahooDownloader(
		start_date="2009-01-01",
		end_date="2021-10-31",
		ticker_list=config_tickers.DOW_30_TICKER,
	).fetch_data()

	print("df.shape = {}.".format(df.shape))
	print('df.sort_values(["date", "tic"], ignore_index=True).head() = {}.'.format(df.sort_values(["date", "tic"], ignore_index=True).head()))

	#--------------------
	# Preprocess data.

	fe = FeatureEngineer(
		use_technical_indicator=True,
		tech_indicator_list=config.INDICATORS,
		use_vix=True,
		use_turbulence=True,
		user_defined_feature = False,
	)

	processed = fe.preprocess_data(df)

	list_ticker = processed["tic"].unique().tolist()
	list_date = list(pd.date_range(processed["date"].min(),processed["date"].max()).astype(str))
	combination = list(itertools.product(list_date, list_ticker))

	processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
	processed_full = processed_full[processed_full["date"].isin(processed["date"])]
	processed_full = processed_full.sort_values(["date", "tic"])

	processed_full = processed_full.fillna(0)

	processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

	#--------------------
	# Build a market environment in OpenAI Gym-style.

	# Data split.
	train = data_split(processed_full, "2009-01-01", "2020-07-01")
	trade = data_split(processed_full, "2020-07-01", "2021-10-31")
	print("len(train) = {}.".format(len(train)))
	print("len(trade) = {}.".format(len(trade)))

	print(train.tail())
	print(trade.head())

	print("config.INDICATORS = {}.".format(config.INDICATORS))

	stock_dimension = len(train.tic.unique())
	state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
	print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

	buy_cost_list = sell_cost_list = [0.001] * stock_dimension
	num_stock_shares = [0] * stock_dimension

	env_kwargs = {
		"hmax": 100,
		"initial_amount": 1000000,
		"num_stock_shares": num_stock_shares,
		"buy_cost_pct": buy_cost_list,
		"sell_cost_pct": sell_cost_list,
		"state_space": state_space,
		"stock_dim": stock_dimension,
		"tech_indicator_list": config.INDICATORS,
		"action_space": stock_dimension,
		"reward_scaling": 1e-4,
	}
	e_train_gym = StockTradingEnv(df=train, **env_kwargs)

	# Environment for training.
	env_train, _ = e_train_gym.get_sb_env()

	print("type(env_train) = {}.".format(type(env_train)))

	#--------------------
	# Train DRL agents.
	# Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC).

	# Agent 1: A2C.
	agent = DRLAgent(env=env_train)
	model_a2c = agent.get_model("a2c")

	trained_a2c = agent.train_model(
		model=model_a2c, 
		tb_log_name="a2c",
		total_timesteps=50000,
	)

	# Agent 2: DDPG.
	agent = DRLAgent(env=env_train)
	model_ddpg = agent.get_model("ddpg")

	trained_ddpg = agent.train_model(
		model=model_ddpg, 
		tb_log_name="ddpg",
		total_timesteps=50000,
	)

	# Agent 3: PPO.
	agent = DRLAgent(env=env_train)
	PPO_PARAMS = {
		"n_steps": 2048,
		"ent_coef": 0.01,
		"learning_rate": 0.00025,
		"batch_size": 128,
	}
	model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

	trained_ppo = agent.train_model(
		model=model_ppo, 
		tb_log_name="ppo",
		total_timesteps=50000,
	)

	# Agent 4: TD3.
	agent = DRLAgent(env=env_train)
	TD3_PARAMS = {
		"batch_size": 100, 
		"buffer_size": 1000000, 
		"learning_rate": 0.001,
	}
	model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

	trained_td3 = agent.train_model(
		model=model_td3, 
		tb_log_name="td3",
		total_timesteps=30000,
	)

	# Agent 5: SAC.
	agent = DRLAgent(env=env_train)
	SAC_PARAMS = {
		"batch_size": 128,
		"buffer_size": 1000000,
		"learning_rate": 0.0001,
		"learning_starts": 100,
		"ent_coef": "auto_0.1",
	}
	model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

	trained_sac = agent.train_model(
		model=model_sac, 
		tb_log_name="sac",
		total_timesteps=60000,
	)

	# In-sample performance.

	# Set turbulence threshold.
	data_risk_indicator = processed_full[(processed_full.date < "2020-07-01") & (processed_full.date >= "2009-01-01")]
	insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=["date"])

	print("insample_risk_indicator.vix.describe() = {}.".format(insample_risk_indicator.vix.describe()))
	print("insample_risk_indicator.vix.quantile(0.996) = {}.".format(insample_risk_indicator.vix.quantile(0.996)))
	print("insample_risk_indicator.turbulence.describe() = {}.".format(insample_risk_indicator.turbulence.describe()))
	print("insample_risk_indicator.turbulence.quantile(0.996) = {}.".format(insample_risk_indicator.turbulence.quantile(0.996)))

	# Trading (Out-of-sample performance).
	#trade = data_split(processed_full, "2020-07-01", "2021-10-31")
	e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold = 70, risk_indicator_col="vix", **env_kwargs)
	#env_trade, obs_trade = e_trade_gym.get_sb_env()

	print(trade.head())

	df_account_value, df_actions = DRLAgent.DRL_prediction(
		model=trained_sac,
		environment=e_trade_gym,
	)

	print("df_account_value.shape = {}.".format(df_account_value.shape))
	print(df_account_value.tail())
	print(df_actions.head())

	#--------------------
	# Backtesting results.

	# BackTestStats.
	print("==============Get Backtest Results===========")
	now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

	perf_stats_all = backtest_stats(account_value=df_account_value)
	perf_stats_all = pd.DataFrame(perf_stats_all)
	perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

	# Baseline stats.
	print("==============Get Baseline Stats===========")
	baseline_df = get_baseline(
		ticker="^DJI", 
		start=df_account_value.loc[0, "date"],
		end=df_account_value.loc[len(df_account_value) - 1, "date"],
	)

	stats = backtest_stats(baseline_df, value_col_name="close")

	print('df_account_value.loc[0, "date"] = {}.'.format(df_account_value.loc[0, "date"]))
	print('df_account_value.loc[len(df_account_value) - 1, "date"] = {}.'.format(df_account_value.loc[len(df_account_value) - 1, "date"]))

	# BackTestPlot.
	print("==============Compare to DJIA===========")
	#%matplotlib inline
	# S&P 500: ^GSPC.
	# Dow Jones Index: ^DJI.
	# NASDAQ 100: ^NDX.
	backtest_plot(
		df_account_value, 
		baseline_ticker="^DJI", 
		baseline_start=df_account_value.loc[0, "date"],
		baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
	)

# Environment for portfolio allocation.
class StockPortfolioEnv(gym.Env):
	"""A single stock trading environment for OpenAI gym.

	Attributes
	----------
		df: DataFrame
			input data
		stock_dim : int
			number of unique stocks
		hmax : int
			maximum number of shares to trade
		initial_amount : int
			start money
		transaction_cost_pct: float
			transaction cost percentage per trade
		reward_scaling: float
			scaling factor for reward, good for training
		state_space: int
			the dimension of input features
		action_space: int
			equals stock dimension
		tech_indicator_list: list
			a list of technical indicator names
		turbulence_threshold: int
			a threshold to control risk aversion
		day: int
			an increment number to control date

	Methods
	-------
	_sell_stock()
		perform sell action based on the sign of the action
	_buy_stock()
		perform buy action based on the sign of the action
	step()
		at each step the agent will return actions, then 
		we will calculate the reward, and return the next observation.
	reset()
		reset the environment
	render()
		use render to return other functions
	save_asset_memory()
		return account value at each time step
	save_action_memory()
		return actions/positions at each time step
	"""
	metadata = {"render.modes": ["human"]}

	def __init__(
		self, 
		df,
		stock_dim,
		hmax,
		initial_amount,
		transaction_cost_pct,
		reward_scaling,
		state_space,
		action_space,
		tech_indicator_list,
		turbulence_threshold=None,
		lookback=252,
		day=0
	):
		#super(StockEnv, self).__init__()
		#money = 10, scope = 1
		self.day = day
		self.lookback=lookback
		self.df = df
		self.stock_dim = stock_dim
		self.hmax = hmax
		self.initial_amount = initial_amount
		self.transaction_cost_pct = transaction_cost_pct
		self.reward_scaling = reward_scaling
		self.state_space = state_space
		self.action_space = action_space
		self.tech_indicator_list = tech_indicator_list

		# Action_space normalization and shape is self.stock_dim.
		self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
		# Shape = (34, 30).
		# covariance matrix + technical indicators.
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space+len(self.tech_indicator_list), self.state_space))

		# Load data from a pandas dataframe.
		self.data = self.df.loc[self.day,:]
		self.covs = self.data["cov_list"].values[0]
		self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)
		self.terminal = False
		self.turbulence_threshold = turbulence_threshold
		# Initalize state: inital portfolio return + individual stock return + individual weights.
		self.portfolio_value = self.initial_amount

		# Memorize portfolio value each step.
		self.asset_memory = [self.initial_amount]
		# Memorize portfolio return each step.
		self.portfolio_return_memory = [0]
		self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
		self.date_memory = [self.data.date.unique()[0]]

	def step(self, actions):
		#print(self.day)
		self.terminal = self.day >= len(self.df.index.unique()) - 1
		#print(actions)

		if self.terminal:
			df = pd.DataFrame(self.portfolio_return_memory)
			df.columns = ["daily_return"]
			plt.plot(df.daily_return.cumsum(), "r")
			plt.savefig("results/cumulative_reward.png")
			plt.close()
			
			plt.plot(self.portfolio_return_memory, "r")
			plt.savefig("results/rewards.png")
			plt.close()

			print("=================================")
			print("begin_total_asset:{}".format(self.asset_memory[0]))
			print("end_total_asset:{}".format(self.portfolio_value))

			df_daily_return = pd.DataFrame(self.portfolio_return_memory)
			df_daily_return.columns = ["daily_return"]
			if df_daily_return["daily_return"].std() != 0:
				sharpe = (252**0.5) * df_daily_return["daily_return"].mean() / df_daily_return["daily_return"].std()
				print("Sharpe: ", sharpe)
			print("=================================")
			
			return self.state, self.reward, self.terminal,{}
		else:
			#print("Model actions: ", actions)
			# Actions are the portfolio weight.
			# Normalize to sum of 1.
			#if (np.array(actions) - np.array(actions).min()).sum() != 0:
			#	norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
			#else:
			#	norm_actions = actions
			weights = self.softmax_normalization(actions) 
			#print("Normalized actions: ", weights)
			self.actions_memory.append(weights)
			last_day_memory = self.data

			# Load next state.
			self.day += 1
			self.data = self.df.loc[self.day,:]
			self.covs = self.data["cov_list"].values[0]
			self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)
			#print(self.state)
			# Calcualte portfolio return.
			# individual stocks' return * weight.
			portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
			# Update portfolio value.
			new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
			self.portfolio_value = new_portfolio_value

			# Save into memory.
			self.portfolio_return_memory.append(portfolio_return)
			self.date_memory.append(self.data.date.unique()[0])
			self.asset_memory.append(new_portfolio_value)

			# The reward is the new portfolio value or end portfolo value.
			self.reward = new_portfolio_value 
			#print("Step reward: ", self.reward)
			#self.reward = self.reward * self.reward_scaling

		return self.state, self.reward, self.terminal, {}

	def reset(self):
		self.asset_memory = [self.initial_amount]
		self.day = 0
		self.data = self.df.loc[self.day,:]
		# Load states.
		self.covs = self.data["cov_list"].values[0]
		self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list], axis=0)
		self.portfolio_value = self.initial_amount
		#self.cost = 0
		#self.trades = 0
		self.terminal = False 
		self.portfolio_return_memory = [0]
		self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
		self.date_memory = [self.data.date.unique()[0]] 
		return self.state

	def render(self, mode="human"):
		return self.state

	def softmax_normalization(self, actions):
		numerator = np.exp(actions)
		denominator = np.sum(np.exp(actions))
		softmax_output = numerator/denominator
		return softmax_output

	def save_asset_memory(self):
		date_list = self.date_memory
		portfolio_return = self.portfolio_return_memory
		#print(len(date_list))
		#print(len(asset_list))
		df_account_value = pd.DataFrame({"date": date_list, "daily_return": portfolio_return})
		return df_account_value

	def save_action_memory(self):
		# date and close price length must match actions length
		date_list = self.date_memory
		df_date = pd.DataFrame(date_list)
		df_date.columns = ["date"]

		action_list = self.actions_memory
		df_actions = pd.DataFrame(action_list)
		df_actions.columns = self.data.tic.values
		df_actions.index = df_date.date
		#df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
		return df_actions

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def get_sb_env(self):
		e = DummyVecEnv([lambda: self])
		obs = e.reset()
		return e, obs

# REF [site] >> https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/1-Introduction/China_A_share_market_tushare.ipynb
#	Quantitative trading in China A stock market with FinRL.
def quantitative_trading_tutorial():
	raise NotImplementedError

# REF [site] >> https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/1-Introduction/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb
#	Deep Reinforcement Learning for Stock Trading from Scratch: Portfolio Allocation.
def portfolio_allocation_tutorial():
	# Create folders.

	if not os.path.exists("./" + config.DATA_SAVE_DIR):
		os.makedirs("./" + config.DATA_SAVE_DIR)
	if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
		os.makedirs("./" + config.TRAINED_MODEL_DIR)
	if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
		os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
	if not os.path.exists("./" + config.RESULTS_DIR):
		os.makedirs("./" + config.RESULTS_DIR)

	#--------------------
	# Download data.

	print("config_tickers.DOW_30_TICKER = {}.".format(config_tickers.DOW_30_TICKER))

	dp = YahooFinanceProcessor()
	df = dp.download_data(
		start_date="2008-01-01",
		end_date="2021-10-31",
		ticker_list=config_tickers.DOW_30_TICKER,
		time_interval="1D",
	)

	print("df.shape = {}.".format(df.shape))
	print(df.head())

	#--------------------
	# Preprocess data.

	fe = FeatureEngineer(
		use_technical_indicator=True,
		use_turbulence=False,
		user_defined_feature=False,
	)

	df = fe.preprocess_data(df)

	print("df.shape = {}.".format(df.shape))
	print(df.head())

	# Add covariance matrix as states.
	df = df.sort_values(["date", "tic"], ignore_index=True)
	df.index = df.date.factorize()[0]

	cov_list = []
	return_list = []

	# Look back is one year.
	lookback = 252
	for i in range(lookback,len(df.index.unique())):
		data_lookback = df.loc[i-lookback:i,:]
		price_lookback=data_lookback.pivot_table(index="date", columns="tic", values="close")
		return_lookback = price_lookback.pct_change().dropna()
		return_list.append(return_lookback)

		covs = return_lookback.cov().values 
		cov_list.append(covs)

	df_cov = pd.DataFrame({"date": df.date.unique()[lookback:], "cov_list": cov_list, "return_list": return_list})
	df = df.merge(df_cov, on="date")
	df = df.sort_values(["date", "tic"]).reset_index(drop=True)

	print("df.shape = {}.".format(df.shape))
	print(df.head())

	#--------------------
	# Design environment.

	# Training data split: 2009-01-01 to 2020-07-01.
	train = data_split(df, "2009-01-01", "2020-07-01")
	#trade = data_split(df, "2020-01-01", config.END_DATE)

	print(df.head())

	# Environment for portfolio allocation.
	stock_dimension = len(train.tic.unique())
	state_space = stock_dimension
	print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

	env_kwargs = {
		"hmax": 100, 
		"initial_amount": 1000000, 
		"transaction_cost_pct": 0.001, 
		"state_space": state_space, 
		"stock_dim": stock_dimension, 
		"tech_indicator_list": config.INDICATORS, 
		"action_space": stock_dimension, 
		"reward_scaling": 1e-4,
	}

	e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)

	env_train, _ = e_train_gym.get_sb_env()
	print("type(env_train) = {}.".format(type(env_train)))

	#--------------------
	# Implement DRL algorithms.

	# Initialize.
	agent = DRLAgent(env=env_train)

	# Model 1: A2C.
	agent = DRLAgent(env=env_train)
	A2C_PARAMS = {
		"n_steps": 5,
		"ent_coef": 0.005,
		"learning_rate": 0.0002,
	}
	model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)

	trained_a2c = agent.train_model(
		model=model_a2c, 
		tb_log_name="a2c",
		total_timesteps=50000,
	)

	#trained_a2c.save("/content/trained_models/trained_a2c.zip")

	# Model 2: PPO.
	agent = DRLAgent(env=env_train)
	PPO_PARAMS = {
		"n_steps": 2048,
		"ent_coef": 0.005,
		"learning_rate": 0.0001,
		"batch_size": 128,
	}
	model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

	trained_ppo = agent.train_model(
		model=model_ppo, 
		tb_log_name="ppo",
		total_timesteps=80000,
	)

	#trained_ppo.save("/content/trained_models/trained_ppo.zip")

	# Model 3: DDPG.
	agent = DRLAgent(env=env_train)
	DDPG_PARAMS = {
		"batch_size": 128,
		"buffer_size": 50000,
		"learning_rate": 0.001,
	}
	model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)

	trained_ddpg = agent.train_model(
		model=model_ddpg, 
		tb_log_name="ddpg",
		total_timesteps=50000,
	)

	#trained_ddpg.save("/content/trained_models/trained_ddpg.zip")

	# Model 4: SAC.
	agent = DRLAgent(env=env_train)
	SAC_PARAMS = {
		"batch_size": 128,
		"buffer_size": 100000,
		"learning_rate": 0.0003,
		"learning_starts": 100,
		"ent_coef": "auto_0.1",
	}
	model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

	trained_sac = agent.train_model(
		model=model_sac, 
		tb_log_name="sac",
		total_timesteps=50000,
	)

	#trained_sac.save("/content/trained_models/trained_sac.zip")

	# Model 5: TD3.
	agent = DRLAgent(env=env_train)
	TD3_PARAMS = {
		"batch_size": 100, 
		"buffer_size": 1000000, 
		"learning_rate": 0.001,
	}
	model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

	trained_td3 = agent.train_model(
		model=model_td3, 
		tb_log_name="td3",
		total_timesteps=30000,
	)

	#trained_td3.save("/content/trained_models/trained_td3.zip")

	# Trading.
	trade = data_split(df, "2020-07-01", "2021-10-31")
	e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)

	print("trade.shape = {}.".format(trade.shape))

	df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)

	print(df_daily_return.head())
	print(df_actions.head())

	#df_daily_return.to_csv("df_daily_return.csv")
	df_actions.to_csv("df_actions.csv")

	#--------------------
	# Backtest our strategy.

	# BackTestStats.
	from pyfolio import timeseries

	DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
	perf_func = timeseries.perf_stats 
	perf_stats_all = perf_func(
		returns=DRL_strat, 
		factor_returns=DRL_strat, 
		positions=None, transactions=None, turnover_denom="AGB",
	)

	print("==============DRL Strategy Stats===========")
	print(perf_stats_all)

	# Baseline stats.
	print("==============Get Baseline Stats===========")
	baseline_df = get_baseline(
		ticker="^DJI", 
		start=df_daily_return.loc[0, "date"],
		end=df_daily_return.loc[len(df_daily_return) - 1, "date"],
	)

	stats = backtest_stats(baseline_df, value_col_name="close")

	# BackTestPlot.
	import pyfolio
	#%matplotlib inline

	baseline_df = get_baseline(
		ticker="^DJI", start=df_daily_return.loc[0, "date"], end="2021-11-01"
	)

	baseline_returns = get_daily_return(baseline_df, value_col_name="close")

	with pyfolio.plotting.plotting_context(font_scale=1.1):
		pyfolio.create_full_tear_sheet(returns=DRL_strat, benchmark_rets=baseline_returns, set_context=False)

	# Min-variance portfolio allocation.
	from pypfopt.efficient_frontier import EfficientFrontier
	from pypfopt import risk_models

	unique_tic = trade.tic.unique()
	unique_trade_date = trade.date.unique()

	print(df.head())

	# Calculate_portfolio_minimum_variance.
	portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
	initial_capital = 1000000
	portfolio.loc[0, unique_trade_date[0]] = initial_capital

	for i in range(len(unique_trade_date) - 1):
		df_temp = df[df.date == unique_trade_date[i]].reset_index(drop=True)
		df_temp_next = df[df.date == unique_trade_date[i+1]].reset_index(drop=True)
		#Sigma = risk_models.sample_cov(df_temp.return_list[0])
		# Calculate covariance matrix.
		Sigma = df_temp.return_list[0].cov()
		# Portfolio allocation.
		ef_min_var = EfficientFrontier(None, Sigma, weight_bounds=(0, 0.1))
		# Minimum variance,
		raw_weights_min_var = ef_min_var.min_volatility()
		# Get weights,
		cleaned_weights_min_var = ef_min_var.clean_weights()

		# Current capital.
		cap = portfolio.iloc[0, i]
		# Current cash invested for each stock.
		current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
		# Current held shares.
		current_shares = list(np.array(current_cash) / np.array(df_temp.close))
		# Next time period price.
		next_price = np.array(df_temp_next.close)
		# next_price * current share to calculate next total account value.
		portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)

	portfolio=portfolio.T
	portfolio.columns = ["account_value"]

	print(portfolio.head())

	a2c_cumpod = (df_daily_return.daily_return + 1).cumprod() - 1
	min_var_cumpod = (portfolio.account_value.pct_change() + 1).cumprod() - 1
	dji_cumpod = (baseline_returns + 1).cumprod() - 1

	# Plotly: DRL, Min-Variance, DJIA.
	from datetime import datetime as dt
	import plotly
	import plotly.graph_objs as go

	time_ind = pd.Series(df_daily_return.date)

	trace0_portfolio = go.Scatter(x=time_ind, y=a2c_cumpod, mode="lines", name="A2C (Portfolio Allocation)")
	trace1_portfolio = go.Scatter(x=time_ind, y=dji_cumpod, mode="lines", name="DJIA")
	trace2_portfolio = go.Scatter(x=time_ind, y=min_var_cumpod, mode="lines", name="Min-Variance")
	#trace3_portfolio = go.Scatter(x=time_ind, y=ddpg_cumpod, mode="lines", name="DDPG")
	#trace4_portfolio = go.Scatter(x=time_ind, y=addpg_cumpod, mode="lines", name="Adaptive-DDPG")
	#trace5_portfolio = go.Scatter(x=time_ind, y=min_cumpod, mode="lines", name="Min-Variance")

	#trace4 = go.Scatter(x=time_ind, y=addpg_cumpod, mode="lines", name="Adaptive-DDPG")
	#trace2 = go.Scatter(x=time_ind, y=portfolio_cost_minv, mode="lines", name="Min-Variance")
	#trace3 = go.Scatter(x=time_ind, y=spx_value, mode="lines", name="SPX")

	fig = go.Figure()
	fig.add_trace(trace0_portfolio)
	fig.add_trace(trace1_portfolio)
	fig.add_trace(trace2_portfolio)
	fig.update_layout(
		legend=dict(
			x=0, y=1,
			traceorder="normal",
			font=dict(family="sans-serif", size=15, color="black"),
			bgcolor="White",
			bordercolor="white",
			borderwidth=2,
		),
	)
	#fig.update_layout(legend_orientation="h")
	fig.update_layout(
		title={
			#"text": "Cumulative Return using FinRL",
			"x": 0.5, "y": 0.85,
			"xanchor": "center", "yanchor": "top"
		}
	)
	# With transaction cost.
	#fig.update_layout(title="Quarterly Trade Date")
	fig.update_layout(
		#margin=dict(l=20, r=20, t=20, b=20),
		paper_bgcolor="rgba(1, 1, 0, 0)",
		plot_bgcolor="rgba(1, 1, 0, 0)",
		#xaxis_title="Date",
		yaxis_title="Cumulative Return",
		xaxis={"type": "date", "tick0": time_ind[0], "tickmode": "linear", "dtick": 86400000.0 * 80}
	)
	fig.update_xaxes(showline=True, linecolor="black", showgrid=True, gridwidth=1, gridcolor="LightSteelBlue", mirror=True)
	fig.update_yaxes(showline=True, linecolor="black", showgrid=True, gridwidth=1, gridcolor="LightSteelBlue", mirror=True)
	fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")

	fig.show()

def main():
	#multiple_stock_trading_tutorial()
	#quantitative_trading_tutorial()  # Not yet implemented.
	portfolio_allocation_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
