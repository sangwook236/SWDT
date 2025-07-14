#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
from ta.utils import dropna

# REF [site] >> https://github.com/bukosabino/ta
def adding_all_features_example():
	from ta import add_all_ta_features

	# Load datas
	df = pd.read_csv("ta/tests/data/datas.csv", sep=",")

	# Clean NaN values
	df = dropna(df)

	# Add all ta features
	df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC")

# REF [site] >> https://github.com/bukosabino/ta
def adding_particular_feature_example():
	from ta.volatility import BollingerBands

	# Load datas
	df = pd.read_csv("ta/tests/data/datas.csv", sep=",")

	# Clean NaN values
	df = dropna(df)

	# Initialize Bollinger Bands Indicator
	indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)

	# Add Bollinger Bands features
	df["bb_bbm"] = indicator_bb.bollinger_mavg()
	df["bb_bbh"] = indicator_bb.bollinger_hband()
	df["bb_bbl"] = indicator_bb.bollinger_lband()

	# Add Bollinger Band high indicator
	df["bb_bbhi"] = indicator_bb.bollinger_hband_indicator()

	# Add Bollinger Band low indicator
	df["bb_bbli"] = indicator_bb.bollinger_lband_indicator()

	# Add Width Size Bollinger Bands
	df["bb_bbw"] = indicator_bb.bollinger_wband()

	# Add Percentage Bollinger Bands
	df["bb_bbp"] = indicator_bb.bollinger_pband()

def main():
	# Install:
	#	pip install ta

	adding_all_features_example()
	adding_particular_feature_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
