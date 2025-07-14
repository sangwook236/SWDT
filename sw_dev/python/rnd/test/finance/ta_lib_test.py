#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import talib

# REF [site] >> https://github.com/TA-Lib/ta-lib-python
def simple_example():
	# List of functions
	print("Functions:")
	print(talib.get_functions())

	# Dict of functions by group
	print("Functions by group:")
	print(talib.get_function_groups())

	#-----
	c = np.random.randn(100)

	k, d = talib.STOCHRSI(c)

	# This produces the same result, calling STOCHF
	rsi = talib.RSI(c)
	k, d = talib.STOCHF(rsi, rsi, rsi)

	# You might want this instead, calling STOCH
	rsi = talib.RSI(c)
	k, d = talib.STOCH(rsi, rsi, rsi)

	#-----
	# Function API

	close = np.random.random(100)

	output = talib.SMA(close)
	upper, middle, lower = talib.BBANDS(close, matype=talib.MA_Type.T3)

	# NaN.
	c = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])
	print(f"{talib.SMA(c, 3)=}.")

	import pandas as pd
	c = pd.Series([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0])
	print(f"{c.rolling(3).mean()=}.")

	#-----
	# Abstract API

	inputs = {
		"open": np.random.random(100),
		"high": np.random.random(100),
		"low": np.random.random(100),
		"close": np.random.random(100),
		"volume": np.random.random(100),
	}

	#SMA = talib.abstract.SMA
	#SMA = talib.abstract.Function("sma")

	# Uses close prices (default)
	output = talib.abstract.SMA(inputs, timeperiod=25)

	# Uses open prices
	output = talib.abstract.SMA(inputs, timeperiod=25, price="open")

	# Uses close prices (default)
	upper, middle, lower = talib.abstract.BBANDS(inputs, 20, 2.0, 2.0)

	# Uses high, low, close (default)
	slowk, slowd = talib.abstract.STOCH(inputs, 5, 3, 0, 3, 0)  # Uses high, low, close by default

	# Uses high, low, open instead
	slowk, slowd = talib.abstract.STOCH(inputs, 5, 3, 0, 3, 0, prices=["high", "low", "open"])

	#-----
	# Streaming API.

	close = np.random.random(100)

	# The Function API
	output = talib.SMA(close)

	# The Streaming API
	latest = talib.stream.SMA(close)

	# The latest value is the same as the last output value
	assert (output[-1] - latest) < 0.00001

# REF [site] >> https://github.com/edgetrader/candlestick-pattern/blob/master/notebook/candlestick-pattern.ipynb
def candlestick_pattern_test():

	raise NotImplementedError

def main():
	# Install:
	#	pip install TA-Lib
	#	conda install -c conda-forge ta-lib

	simple_example()  # Not yet tested

	# Candlestick patterns
	#candlestick_pattern_test()  # Not yet implemented

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
