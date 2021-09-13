#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import stockstats
import yfinance

# REF [site] >> https://github.com/jealous/stockstats
def simple_example():
	df = yfinance.download(
		"AAPL",
		start="2020-01-01",
		end="2020-12-31"
	)

	#--------------------
	stock_df = stockstats.StockDataFrame.retype(df)

	"""
	Supported statistics/indicators are:

	change (in percent)
	delta
	permutation (zero based)
	log return
	max in range
	min in range
	middle = (close + high + low) / 3
	compare: le, ge, lt, gt, eq, ne
	count: both backward(c) and forward(fc)
	SMA: simple moving average
	EMA: exponential moving average
	MSTD: moving standard deviation
	MVAR: moving variance
	RSV: raw stochastic value
	RSI: relative strength index
	KDJ: Stochastic oscillator
	Bolling: including upper band and lower band.
	MACD: moving average convergence divergence. Including signal and histogram. (see note)
	CR:
	WR: Williams Overbought/Oversold index
	CCI: Commodity Channel Index
	TR: true range
	ATR: average true range
	line cross check, cross up or cross down.
	DMA: Different of Moving Average (10, 50)
	DMI: Directional Moving Index, including
		+DI: Positive Directional Indicator
		-DI: Negative Directional Indicator
		ADX: Average Directional Movement Index
		ADXR: Smoothed Moving Average of ADX
	TRIX: Triple Exponential Moving Average
	TEMA: Another Triple Exponential Moving Average
	VR: Volatility Volume Ratio
	MFI: Money Flow Index
	"""

	"""
	# Volume delta against previous day.
	stock_df['volume_delta']

	# Open delta against next 2 day.
	stock_df['open_2_d']

	# Open price change (in percent) between today and the day before yesterday 'r' stands for rate.
	stock_df['open_-2_r']

	# CR indicator, including 5, 10, 20 days moving average.
	stock_df['cr']
	stock_df['cr-ma1']
	stock_df['cr-ma2']
	stock_df['cr-ma3']

	# Volume max of three days ago, yesterday and two days later.
	stock_df['volume_-3,2,-1_max']

	# Volume min between 3 days ago and tomorrow.
	stock_df['volume_-3~1_min']

	# KDJ (Stochastic oscillator), default to 9 days.
	stock_df['kdjk']
	stock_df['kdjd']
	stock_df['kdjj']

	# Three days KDJK cross up 3 days KDJD.
	stock_df['kdj_3_xu_kdjd_3']

	# 2 days simple moving average on open price.
	stock_df['open_2_sma']

	# MACD (Moving Average Convergence Divergence).
	stock_df['macd']
	# MACD signal line.
	stock_df['macds']
	# MACD histogram.
	stock_df['macdh']

	# Bolling, including upper band and lower band.
	stock_df['boll']
	stock_df['boll_ub']
	stock_df['boll_lb']

	# Close price less than 10.0 in 5 days count.
	stock_df['close_10.0_le_5_c']

	# CR MA2 cross up CR MA1 in 20 days count.
	stock_df['cr-ma2_xu_cr-ma1_20_c']

	# Count forward(future) where close price is larger than 10.
	stock_df['close_10.0_ge_5_fc']

	# 6 days RSI (Relative Strength Index).
	stock_df['rsi_6']
	# 12 days RSI.
	stock_df['rsi_12']

	# 10 days WR (Williams Overbought/Oversold index).
	stock_df['wr_10']
	# 6 days WR.
	stock_df['wr_6']

	# CCI, default to 14 days.
	stock_df['cci']
	# 20 days CCI.
	stock_df['cci_20']

	# TR (true range).
	stock_df['tr']
	# ATR (Average True Range).
	stock_df['atr']

	# DMA, difference of 10 and 50 moving average.
	stock_df['dma']

	# DMI.
	# +DI, default to 14 days.
	stock_df['pdi']
	# -DI, default to 14 days.
	stock_df['mdi']
	# DX, default to 14 days of +DI and -DI.
	stock_df['dx']
	# ADX, 6 days SMA of DX, same as stock_df['dx_6_ema'].
	stock_df['adx']
	# ADXR, 6 days SMA of ADX, same as stock_df['adx_6_ema'].
	stock_df['adxr']

	# TRIX, default to 12 days.
	stock_df['trix']
	# TRIX based on the close price for a window of 3.
	stock_df['close_3_trix']
	# MATRIX is the simple moving average of TRIX.
	stock_df['trix_9_sma']
	# TEMA, another implementation for triple ema.
	stock_df['tema']
	# TEMA based on the close price for a window of 2.
	stock_df['close_2_tema']

	# VR, default to 26 days.
	stock_df['vr']
	# MAVR is the simple moving average of VR.
	stock_df['vr_6_sma']

	# Money flow index, default to 14 days.
	stock_df['mfi']
	"""

	print(stock_df.columns.values)
	print(stock_df.head())

	print(stock_df[["change", "rate", "close_-1_d", "log-ret"]])
	print(stock_df[["close_10_sma"]])

	stock_df[["close", "close_10_sma", "close_50_sma"]].plot(title="SMA example")
	stock_df.loc["2020-06-01":, ["close", "close_10_sma", "close_50_sma"]].plot(title="SMA example")
	stock_df[["close", "boll", "boll_ub", "boll_lb"]].plot(title="Bollinger Bands example")

def main():
	simple_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
