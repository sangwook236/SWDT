#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import stockstats

# REF [site] >> https://github.com/jealous/stockstats
def simple_example():
	if True:
		import yfinance as yf

		df = yf.download(
			"AAPL",
			start="2020-01-01",
			end="2020-12-31"
		)
		sdf = stockstats.StockDataFrame.retype(df, index_column=None)
	else:
		import pandas as pd

		df = pd.read_csv("./stock.csv")
		sdf = stockstats.wrap(df, index_column=None)

		#df = stockstats.unwrap(sdf)

	#--------------------
	"""
	Supported statistics/indicators:

	delta
	permutation (zero-based)
	log return
	max in range
	min in range
	middle = (close + high + low) / 3
	compare: le, ge, lt, gt, eq, ne
	count: both backward(c) and forward(fc)
	cross: including upward cross and downward cross
	SMA: Simple Moving Average
	EMA: Exponential Moving Average
	MSTD: Moving Standard Deviation
	MVAR: Moving Variance
	RSV: Raw Stochastic Value
	RSI: Relative Strength Index
	KDJ: Stochastic Oscillator
	Bolling: Bollinger Band
	MACD: Moving Average Convergence Divergence
	CR: Energy Index (Intermediate Willingness Index)
	WR: Williams Overbought/Oversold index
	CCI: Commodity Channel Index
	TR: True Range
	ATR: Average True Range
	DMA: Different of Moving Average (10, 50)
	DMI: Directional Moving Index, including
		+DI: Positive Directional Indicator
		-DI: Negative Directional Indicator
		ADX: Average Directional Movement Index
		ADXR: Smoothed Moving Average of ADX
	TRIX: Triple Exponential Moving Average
	TEMA: Another Triple Exponential Moving Average
	VR: Volume Variation Index
	MFI: Money Flow Index
	VWMA: Volume Weighted Moving Average
	CHOP: Choppiness Index
	KER: Kaufman's efficiency ratio
	KAMA: Kaufman's Adaptive Moving Average
	PPO: Percentage Price Oscillator
	StochRSI: Stochastic RSI
	WT: LazyBear's Wave Trend
	Supertrend: with the Upper Band and Lower Band
	Aroon: Aroon Oscillator
	Z: Z-Score
	AO: Awesome Oscillator
	BOP: Balance of Power
	MAD: Mean Absolute Deviation
	ROC: Rate of Change
	Coppock: Coppock Curve
	Ichimoku: Ichimoku Cloud
	CTI: Correlation Trend Indicator
	LRMA: Linear Regression Moving Average
	ERI: Elder-Ray Index
	FTR: the Gaussian Fisher Transform Price Reversals indicator
	RVGI: Relative Vigor Index
	Inertia: Inertia Indicator
	KST: Know Sure Thing
	PGO: Pretty Good Oscillator
	PSL: Psychological Line
	PVO: Percentage Volume Oscillator
	QQE: Quantitative Qualitative Estimation
	"""

	"""
	# Volume delta against previous day.
	sdf["volume_delta"]

	# Open delta against next 2 day.
	sdf["open_2_d"]

	# Open price change (in percent) between today and the day before yesterday 'r' stands for rate.
	sdf["open_-2_r"]

	# CR indicator, including 5, 10, 20 days moving average.
	sdf["cr"]
	sdf["cr-ma1"]
	sdf["cr-ma2"]
	sdf["cr-ma3"]

	# Volume max of three days ago, yesterday and two days later.
	sdf["volume_-3,2,-1_max"]

	# Volume min between 3 days ago and tomorrow.
	sdf["volume_-3~1_min"]

	# KDJ (Stochastic oscillator), default to 9 days.
	sdf["kdjk"]
	sdf["kdjd"]
	sdf["kdjj"]

	# Three days KDJK cross up 3 days KDJD.
	sdf["kdj_3_xu_kdjd_3"]

	# 2 days simple moving average on open price.
	sdf["open_2_sma"]

	# MACD (Moving Average Convergence Divergence).
	sdf["macd"]
	# MACD signal line.
	sdf["macds"]
	# MACD histogram.
	sdf["macdh"]

	# Bolling, including upper band and lower band.
	sdf["boll"]
	sdf["boll_ub"]
	sdf["boll_lb"]

	# Close price less than 10.0 in 5 days count.
	sdf["close_10.0_le_5_c"]

	# CR MA2 cross up CR MA1 in 20 days count.
	sdf["cr-ma2_xu_cr-ma1_20_c"]

	# Count forward(future) where close price is larger than 10.
	sdf["close_10.0_ge_5_fc"]

	# 6 days RSI (Relative Strength Index).
	sdf["rsi_6"]
	# 12 days RSI.
	sdf["rsi_12"]

	# 10 days WR (Williams Overbought/Oversold index).
	sdf["wr_10"]
	# 6 days WR.
	sdf["wr_6"]

	# CCI, default to 14 days.
	sdf["cci"]
	# 20 days CCI.
	sdf["cci_20"]

	# TR (true range).
	sdf["tr"]
	# ATR (Average True Range).
	sdf["atr"]

	# DMA, difference of 10 and 50 moving average.
	sdf["dma"]

	# DMI.
	# +DI, default to 14 days.
	sdf["pdi"]
	# -DI, default to 14 days.
	sdf["mdi"]
	# DX, default to 14 days of +DI and -DI.
	sdf["dx"]
	# ADX, 6 days SMA of DX, same as sdf["dx_6_ema"].
	sdf["adx"]
	# ADXR, 6 days SMA of ADX, same as sdf["adx_6_ema"].
	sdf["adxr"]

	# TRIX, default to 12 days.
	sdf["trix"]
	# TRIX based on the close price for a window of 3.
	sdf["close_3_trix"]
	# MATRIX is the simple moving average of TRIX.
	sdf["trix_9_sma"]
	# TEMA, another implementation for triple ema.
	sdf["tema"]
	# TEMA based on the close price for a window of 2.
	sdf["close_2_tema"]

	# VR, default to 26 days.
	sdf["vr"]
	# MAVR is the simple moving average of VR.
	sdf["vr_6_sma"]

	# Money flow index, default to 14 days.
	sdf["mfi"]
	"""

	print(f"{sdf.columns.values=}.")
	print(sdf.head())

	print(sdf[["change", "rate", "close_-1_d", "log-ret"]])
	print(sdf[["close_10_sma"]])

	# Plot.
	sdf[["close", "close_10_sma", "close_50_sma"]].plot(title="SMA example")
	sdf.loc["2020-06-01":, ["close", "close_10_sma", "close_50_sma"]].plot(title="SMA example")
	sdf[["close", "boll", "boll_ub", "boll_lb"]].plot(title="Bollinger Bands example")

def main():
	simple_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
