#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import yfinance as yf
import matplotlib.pyplot as plt

def explicit_heat_smooth(prices: np.array, t_end: float = 3.0) -> np.array:
	'''
	Smoothen out a time series using a simple explicit finite difference method.
	The scheme uses a first-order method in time, and a second-order centred
	difference approximation in space. The scheme is only numerically stable
	if the time-step 0<=k<=1.

	The prices are fixed at the end-points, so the interior is smoothed.

	Parameters
	----------
	prices : np.array
		The price to smoothen
	t_end : float
		The time at which to terminate the smootheing (i.e. t = 2)

	Returns
	-------
	P : np.array
		The smoothened time-series
	'''

	k = 0.1  # Time spacing.

	# Set up the initial condition.
	P = prices

	t = 0
	while t < t_end:
		# Solve the finite difference scheme for the next time-step.
		P = k * (P[2:] + P[:-2]) + P[1:-1] * (1 - 2 * k)

		# Add the fixed boundary conditions since the above solves the interior points only.
		P = np.hstack((np.array([prices[0]]), P, np.array([prices[-1]])))
		t += k

	return P

def heat_analytical_smooth(prices: np.array, t: float = 3.0, m: int = 200) -> np.array:
	'''
	Find the analytical solution to the heat equation

	See: https://tutorial.math.lamar.edu/classes/de/heateqnnonzero.aspx

	Parameters
	----------
	prices : np.array
		The price to smoothen.
	t : float
		The time at which to terminate the smootheing (i.e. t = 2)
	m : int
		The amount of terms in the solution's Fourier series

	Returns
	-------
	np.array
		The analytical solution to the heat equation
	'''

	p0 = prices[0]
	pn = prices[-1]

	n = prices.shape[0]
	x = np.arange(0, n, dtype=np.float32)
	M = np.arange(1, m, dtype=np.float32)

	L = n - 1
	u_e = p0 + (pn - p0) * x / L

	mx = M.reshape(-1, 1)@x.reshape(1, -1)
	sin_m_pi_x = np.sin(mx * np.pi / L)

	# Calculate the B_m terms using numerical quadrature (trapezium rule).
	bm = 2 * np.sum((sin_m_pi_x * (prices - u_e)).T, axis=0) / n

	return u_e + np.sum((bm * np.exp(-t * (M * np.pi / L)**2)).reshape(-1, 1) * sin_m_pi_x, axis=0)

# REF [site] >> https://medium.com/geekculture/a-surprising-way-to-smoothen-a-time-series-solving-the-heat-equation-c73082dd9cd7
def time_series_smoothing_1():
	df = yf.download('TSLA')
	df = df[-100:]

	df.loc[:, 'close_smooth1'] = explicit_heat_smooth(df['Close'].values, t_end=3)
	df.loc[:, 'close_smooth2'] = heat_analytical_smooth(df['Close'].values, t=3, m=200)

	df.plot(
		y=['Close', 'close_smooth1', 'close_smooth2'],
		xlabel='Date',
		ylabel='Price',
		title='TSLA Stock Price (Closing Prices)',
	)
	plt.show()

# REF [site] >> https://medium.com/geekculture/a-surprising-way-to-smoothen-a-time-series-solving-the-heat-equation-c73082dd9cd7
def time_series_smoothing_2():
	warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

	def smooth_heat(vals_to_smooth: np.array):
		for n in range(vals_to_smooth.shape[0]):
			vals_to_smooth[n, :] = explicit_heat_smooth(vals_to_smooth[n, :], t_end=3)
		return vals_to_smooth

	def savgol_smooth(vals_to_smooth: np.array):
		for n in range(vals_to_smooth.shape[0]):
			vals_to_smooth[n, :] = savgol_filter(vals_to_smooth[n, :], 11, 5)
		return vals_to_smooth

	def get_time_series(df, col, name, lags):
		'''
		Get a time series of data and place them as new cols
		'''
		return df.assign(**{f'{name}_t-{lag}': df[col].shift(lag) for lag in lags})

	df = yf.download('AAPL')
	df = get_time_series(df, 'Close', 'close', range(100)).dropna().drop(columns=['Open', 'Low', 'High', 'Close', 'Adj Close', 'Volume'])

	vals_to_smooth = df.values

	start_time = time.time()
	heat_eqn1 = smooth_heat(vals_to_smooth)
	print('Heat equation smooth time: {}.'.format(time.time() - start_time))

	start_time = time.time()
	heat_eqn2 = savgol_smooth(vals_to_smooth)
	print('Savitzky Golay smooth time: {}.'.format(time.time() - start_time))

	plt.figure()
	plt.plot(vals_to_smooth[10000:, 0], label='Raw')
	plt.plot(heat_eqn1[10000:, 0], label='Heat')
	plt.plot(heat_eqn2[10000:, 0], label='Savgol')
	plt.legend()

	plt.show()

def main():
	time_series_smoothing_1()
	#time_series_smoothing_2()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
