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
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl import config
from finrl import config_tickers

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
	combination = list(itertools.product(list_date,list_ticker))

	processed_full = pd.DataFrame(combination,columns=["date", "tic"]).merge(processed,on=["date","tic"],how="left")
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

	trained_ppo = agent.train_model(
		model=model_ppo, 
		tb_log_name="ppo",
		total_timesteps=50000,
	)

	model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

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
		environment=e_trade_gym
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
		end=df_account_value.loc[len(df_account_value)-1, "date"]
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
		baseline_end=df_account_value.loc[len(df_account_value)-1, "date"]
	)

def main():
	multiple_stock_trading_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
