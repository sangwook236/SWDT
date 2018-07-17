#!/usr/bin/env python

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import math

# REF [site] >> https://kr.mathworks.com/help/matlab/ref/fft.html
def generate_toy_signal_1(time, noise=True, DC=True):
	sig_amp, sig_freq = 0.7, 50  # Amplitude and frequency of the signal.
	signal = sig_amp * np.sin(2 * np.pi * sig_freq * time)
	sig_amp, sig_freq = 1.0, 120  # Amplitude and frequency of the signal.
	signal += sig_amp * np.sin(2 * np.pi * sig_freq * time)

	if noise:
		# mu = 0, std = 2.
		signal += 2 * np.random.randn(*time.shape);
	if DC:
		signal += 2;

	return signal

def generate_toy_signal_2(time, noise=True, DC=True):
	amp, freq = 1.0, 50  # Amplitude and frequency of the signal.
	signal = amp * np.sin(2 * np.pi * freq * time)
	amp, freq = 1.0, 70  # Amplitude and frequency of the signal.
	signal += amp * np.sin(2 * np.pi * freq * time)
	amp, freq = 1.0, 150  # Amplitude and frequency of the signal.
	signal += amp * np.sin(2 * np.pi * freq * time)
	amp, freq = 1.0, 300  # Amplitude and frequency of the signal.
	signal += amp * np.sin(2 * np.pi * freq * time)
	amp, freq = 1.0, 350  # Amplitude and frequency of the signal.
	signal += amp * np.sin(2 * np.pi * freq * time)

	if noise:
		# mu = 0, std = 2.
		signal += 2 * np.random.randn(*time.shape);
	if DC:
		signal += 2;

	return signal

def next_power_of_2(x):
	return 2**math.ceil(math.log2(x)) if x > 0 else None

# REF [site] >> https://kr.mathworks.com/help/matlab/ref/fft.html
def compute_fft(signal, Fs):
	N = signal.shape[0]
	#NFFT = N
	NFFT = next_power_of_2(N)
	NFFT_2 = int(NFFT / 2) + 1

	freq = np.linspace(0, 1, NFFT_2) * Fs / 2

	# Double-sided spectrum: [-Fs/2, Fs/2).
	#sig_fft = fftpack.fft(signal, NFFT)
	# Parseval's theorem of DFT:
	#	The energy of the time domain signal is equal to the energy of the frequency domain signal divided by the lenght of the sequence N.
	sig_fft = fftpack.fft(signal, NFFT) / N

	# Single-sided spectrum: [0, Fs/2).
	sig_fft = sig_fft[:NFFT_2]
	# Scale power.
	#	By definition, the area underneath the curve is the total power or variance of the function (depending on your domain).
	#	This is true whether you are looking at the double-sided or single-sided amplitude.
	#	The single-sided amplitude is the positive half of the double-sided one.
	sig_fft[1:NFFT_2-1] = 2 * sig_fft[1:NFFT_2-1]

	return sig_fft, freq

def plot_fft(signal, Fs):
	sig_fft, freq = compute_fft(signal, Fs)
	N = signal.shape[0]
	time = np.arange(N) / float(Fs)

	fig, ax = plt.subplots(2, 1)
	ax[0].plot(time, signal, 'r')
	ax[0].set_title('Waveform')
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Amplitude')
	ax[1].plot(freq, abs(sig_fft))
	#ax[1].plot(freq[1:], abs(sig_fft[1:]))  # Removes DC (0 Hz) response.
	ax[1].set_title('Single-sided Amplitude Spectrum (FFT)')
	ax[1].set_xlabel('Freq (Hz)')
	ax[1].set_ylabel('Magnitude')
