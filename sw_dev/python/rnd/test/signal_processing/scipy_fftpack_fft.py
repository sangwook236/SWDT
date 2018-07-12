#!/usr/bin/env python

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import traceback, sys

def toy_example():
	Fs = 100
	N = 100

	N_2 = int(N / 2) + 1
	time = np.arange(N) / float(Fs)
	freq = np.linspace(0, 1, N_2) * Fs / 2

	sig_freq = 5  # Frequency of the signal.
	signal = np.sin(2 * np.pi * sig_freq * time)
	sig_freq = 7  # Frequency of the signal.
	signal += np.sin(2 * np.pi * sig_freq * time)
	sig_freq = 15  # Frequency of the signal.
	signal += np.sin(2 * np.pi * sig_freq * time)
	sig_freq = 30  # Frequency of the signal.
	signal += np.sin(2 * np.pi * sig_freq * time)

	#signal_fft = fftpack.fft(signal)
	signal_fft = fftpack.fft(signal) / N
	signal_fft = signal_fft[:N_2]

	fig, ax = plt.subplots(2, 1)
	ax[0].plot(time, signal)
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Amplitude')
	ax[1].plot(freq, abs(signal_fft), 'r')
	ax[1].set_xlabel('Freq (Hz)')
	ax[1].set_ylabel('Magnitude')

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
