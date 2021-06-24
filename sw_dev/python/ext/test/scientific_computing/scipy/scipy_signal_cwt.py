#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def scalogram_example_1():
	if True:
		#N = 30
		N = 1000
		t = np.linspace(-1, 1, N, endpoint=False)
		sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
	else:
		# A signal given.
		sig = ...
		N = len(sig)
		t = np.linspace(0, 1 / Fs * N, N + 1)[:N]
	widths = np.arange(1, N + 1)
	#widths = np.arange(1, 500 + 1)

	# Compute the CWT.
	cwtmatr = signal.cwt(sig, signal.ricker, widths)  # cwtmatr.shape = (len(widths), len(sig)).

	# Scalogram: a spectrogram for wavelets. (???)
	plt.figure()
	plt.pcolormesh(t, widths, 20 * np.log10(np.abs(cwtmatr)))
	#plt.pcolormesh(t, widths[:500], 20 * np.log10(np.abs(cwtmatr[:500,:])))
	plt.title('Scalogram')
	#plt.xlabel('?')
	#plt.ylabel('?')
	plt.tight_layout()

	# Plot the CWT.
	plt.figure()
	#plt.imshow(cwtmatr, extent=[t[0], t[-1], widths[0], widths[-1]], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
	plt.imshow(cwtmatr, extent=[t[0], t[-1], widths[-1], widths[0]], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
	#plt.imshow(cwtmatr[:200,:], extent=[t[0], t[-1], widths[200], widths[0]], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
	plt.title('CWT')
	#plt.xlabel('?')
	#plt.ylabel('?')
	plt.tight_layout()

	plt.show()

#--------------------------------------------------------------------

# REF [site] >> https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.spectrogram.html

def scalogram_example_2():
	# Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.
	fs = 10e3
	N = 1e5

	amp = 2 * np.sqrt(2)
	noise_power = 0.01 * fs / 2
	time = np.arange(N) / float(fs)
	mod = 500 * np.cos(2 * np.pi * 0.25 * time)
	carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
	noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
	noise *= np.exp(-time / 5)
	x = carrier + noise
	# TODO [fix] >> How to determine?
	scale = np.arange(1, 101)

	# Compute the CWT.
	cwtmatr = signal.cwt(x, signal.ricker, scale)

	# Scalogram: a spectrogram for wavelets. (???)
	plt.pcolormesh(time, scale, 20 * np.log10(np.abs(cwtmatr)))
	plt.title('Scalogram')
	#plt.xlabel('?')
	#plt.ylabel('?')
	plt.tight_layout()

	# Plot the CWT.
	plt.imshow(cwtmatr, extent=[0, 10, 30, -30], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
	plt.title('CWT')
	#plt.xlabel('?')
	#plt.ylabel('?')
	plt.tight_layout()

	plt.show()

def main():
	scalogram_example_1()
	scalogram_example_2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
