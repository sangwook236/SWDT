#!/usr/bin/env python

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

import numpy as np
import math, scipy.stats
import traceback, sys

def compute_rms(signal, axis=0, is_centered=True):
	return np.sqrt(np.mean((signal - np.mean(signal, axis=axis))**2, axis=axis)) if is_centered else np.sqrt(np.mean(signal**2, axis=axis))

def compute_peak(signal, axis=0):
	return np.max(np.abs(signal), axis=axis)

def compute_peak2peak(signal, axis=0):
	#return np.ptp(signal, axis=axis)
	return np.max(signal, axis=axis) - np.min(signal, axis=axis)

def compute_crest_factor(signal, axis=0):
	rms = compute_rms(signal, axis=axis, is_centered=False)
	peak = compute_peak(signal, axis=axis)
	cf = np.zeros_like(rms)
	for idx, (r, p) in enumerate(zip(rms, peak)):
		cf[idx] = np.inf if np.isclose(r, 0.0) else p / r
	return cf

def compute_skewness(signal, axis=0):
	return scipy.stats.skew(signal, axis=axis, bias=True)

def compute_kurtosis(signal, axis=0):
	return scipy.stats.kurtosis(signal, axis=axis, fisher=False, bias=True)

def basic_statistics():
	if False:
		time = np.linspace(0.0, 1.0, num=1000)
		signal = np.cos(2 * math.pi * 100 * time)
		signal = np.vstack([signal, signal * 2, signal * 3, signal * 4]).T
	else:
		signal = np.array(
			[[1.1650,  1.6961, -1.4462, -0.3600],
			[0.6268,  0.0591, -0.7012, -0.1356],
			[0.0751,  1.7971,  1.2460, -1.3493],
			[0.3516,  0.2641, -0.6390, -1.2704],
			[-0.6965,  0.8717,  0.5774,  0.9846]]
		)

	print('RMS =', compute_rms(signal))
	print('Peak =', compute_peak(signal))
	print('Peak-to-peak =', compute_peak2peak(signal))
	print('Crest factor =', compute_crest_factor(signal))
	print('Skewness =', compute_skewness(signal))
	print('Kurtosis =', compute_kurtosis(signal))

def main():
	basic_statistics()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	try:
		main()
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
