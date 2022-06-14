#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
def periodogram_example():
	rng = np.random.default_rng()

	# Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz.
	fs = 10e3
	N = 1e5
	amp = 2 * np.sqrt(2)
	freq = 1234.0
	noise_power = 0.001 * fs / 2
	time = np.arange(N) / fs
	x = amp * np.sin(2 * np.pi * freq * time)
	x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

	# Compute and plot the power spectral density.
	f, Pxx_den = signal.periodogram(x, fs)

	plt.figure()
	plt.semilogy(f, Pxx_den)
	plt.ylim([1e-7, 1e2])
	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')
	plt.tight_layout()

	# If we average the last half of the spectral density, to exclude the peak, we can recover the noise power on the signal.
	print('Noise power = {}.'.format(np.mean(Pxx_den[25000:])))

	#--------------------
	# Compute and plot the power spectrum.
	f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')

	plt.figure()
	plt.semilogy(f, np.sqrt(Pxx_spec))
	plt.ylim([1e-4, 1e1])
	plt.xlabel('frequency [Hz]')
	plt.ylabel('Linear spectrum [V RMS]')

	# The peak height in the power spectrum is an estimate of the RMS amplitude.
	print('The estimate of the RMS amplitud = {}.'.format(np.sqrt(Pxx_spec.max())))

	plt.show()

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.htm
def welch_example():
	rng = np.random.default_rng()

	# Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz.
	fs = 10e3
	N = 1e5
	amp = 2 * np.sqrt(2)
	freq = 1234.0
	noise_power = 0.001 * fs / 2
	time = np.arange(N) / fs
	x = amp * np.sin(2 * np.pi * freq * time)
	x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

	# Compute and plot the power spectral density.
	f, Pxx_den = signal.welch(x, fs, nperseg=1024)

	plt.figure()
	plt.semilogy(f, Pxx_den)
	plt.ylim([0.5e-3, 1])
	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')

	# If we average the last half of the spectral density, to exclude the peak, we can recover the noise power on the signal.
	print('Noise power = {}.'.format(np.mean(Pxx_den[256:])))

	#--------------------
	# Compute and plot the power spectrum..
	f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')

	plt.figure()
	plt.semilogy(f, np.sqrt(Pxx_spec))
	plt.xlabel('frequency [Hz]')
	plt.ylabel('Linear spectrum [V RMS]')

	# The peak height in the power spectrum is an estimate of the RMS amplitude.
	print('The estimate of the RMS amplitud = {}.'.format(np.sqrt(Pxx_spec.max())))

	# If we now introduce a discontinuity in the signal,
	# by increasing the amplitude of a small portion of the signal by 50,
	# we can see the corruption of the mean average power spectral density,
	# but using a median average better estimates the normal behaviour.
	x[int(N // 2):int(N // 2) + 10] *= 50.

	f, Pxx_den = signal.welch(x, fs, nperseg=1024)
	f_med, Pxx_den_med = signal.welch(x, fs, nperseg=1024, average='median')

	plt.figure()
	plt.semilogy(f, Pxx_den, label='mean')
	plt.semilogy(f_med, Pxx_den_med, label='median')
	plt.ylim([0.5e-3, 1])
	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')
	plt.legend()

	plt.show()

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html
def csd_example():
	rng = np.random.default_rng()

	# Generate two test signals with some common features.
	fs = 10e3
	N = 1e5
	amp = 20
	freq = 1234.0
	noise_power = 0.001 * fs / 2
	time = np.arange(N) / fs
	b, a = signal.butter(2, 0.25, 'low')
	x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
	y = signal.lfilter(b, a, x)
	x += amp * np.sin(2 * np.pi * freq * time)
	y += rng.normal(scale=0.1 * np.sqrt(noise_power), size=time.shape)

	# Compute and plot the magnitude of the cross spectral density.
	f, Pxy = signal.csd(x, y, fs, nperseg=1024)

	plt.semilogy(f, np.abs(Pxy))
	plt.xlabel('frequency [Hz]')
	plt.ylabel('CSD [V**2/Hz]')

	plt.show()

def main():
	#periodogram_example()
	#welch_example()
	csd_example()  # Cross spectral density.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
