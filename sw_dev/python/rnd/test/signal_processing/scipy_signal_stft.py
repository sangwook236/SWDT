#!/usr/bin/env python

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
def stft_example():
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

	# Compute the spectrogram.
	#f, t, Zxx = signal.stft(x, fs, nperseg=256)  # Zxx.shape = (129, 783).
	f, t, Zxx = signal.stft(x, fs, nperseg=1000)  # Zxx.shape = (501, 201).

	# Plot the spectrogram.
	#plt.pcolormesh(t, f, np.abs(Zxx))
	plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)))
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

def main():
	stft_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
