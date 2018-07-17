#!/usr/bin/env python

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

import numpy as np
import traceback, sys
import fft_util

def toy_example():
	Fs = 1000
	N = 1500
	time = np.arange(N) / float(Fs)

	#signal = generate_toy_signal_1(time, noise=False, DC=True)
	signal = fft_util.generate_toy_signal_2(time, noise=False, DC=True)

	fft_util.plot_fft(signal, Fs)
	#fft_util.plot_fft(signal - signal.mean(), Fs)  # Removes DC component.

def main():
	toy_example()

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
