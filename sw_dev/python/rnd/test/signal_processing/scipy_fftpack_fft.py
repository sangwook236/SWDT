#!/usr/bin/env python

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

import numpy as np
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

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
