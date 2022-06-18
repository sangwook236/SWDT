#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def fft_example():
	t = np.arange(256)
	freq = np.fft.fftfreq(t.shape[-1])

	S = np.fft.fft(np.sin(t))

	fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True)
	ax[0].plot(freq, S.real)
	ax[0].set(title='Real', xlabel='Frequency')
	ax[1].plot(freq, S.imag)
	ax[1].set(title='Imaginary', xlabel='Frequency')
	plt.tight_layout()

	#--------------------
	t = np.arange(400)
	n = np.zeros((400,), dtype=complex)
	n[40:60] = np.exp(1j * np.random.uniform(0, 2 * np.pi, (20,)))

	s = np.fft.ifft(n)

	fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True)
	ax[0].plot(t, s.real)
	ax[0].set(title='Real', xlabel='Time')
	ax[1].plot(t, s.imag)
	ax[1].set(title='Imaginary', xlabel='Time')

	#--------------------
	if True:
		t = np.arange(1024)
		y = 12 * np.sin(t) + 20 * np.sin(10 * t) + 7 * np.sin(25 * t) + np.random.randn(t.shape[-1])
	else:
		import librosa
		y, sr = librosa.load(librosa.ex('trumpet'))
		t = np.arange(len(y)) / sr

	S = np.fft.fft(y)
	y_hat = np.fft.ifft(S)

	S_mag = np.abs(S)
	y_mag_hat = np.fft.ifft(S_mag)
	S_phase = np.exp(1.0j * np.angle(S))
	y_phase_hat = np.fft.ifft(S_phase)

	fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
	ax[0, 0].plot(t, y)
	ax[0, 0].set(title='$y$')
	ax[0, 1].plot(t, y_hat)
	ax[0, 1].set(title='$\hat{y}$')
	ax[1, 0].plot(t, y_mag_hat)
	ax[1, 0].set(title='$\hat{y}_{mag}$')
	ax[1, 1].plot(t, y_phase_hat)
	ax[1, 1].set(title='$\hat{y}_{phase}$')
	plt.tight_layout()

	plt.show()

def main():
	fft_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
