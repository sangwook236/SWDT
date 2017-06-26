# A spectrogram, sonogram, spectral waterfalls, voiceprints, or voicegrams: a visual representation of the spectrum of frequencies in a sound.

#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/sangwook/my_dataset"
#dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
dataset_home_dir_path = "D:/dataset"

data_dir_path = dataset_home_dir_path + "/failure_analysis/defect/motor_20170621/0_original/500-1500Hz"

#%%------------------------------------------------------------------

# REF [site] >> https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.spectrogram.html

from scipy import signal
import matplotlib.pyplot as plt

# Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

fs = 10e3
N = 1e5

amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500 * np.cos(2 * np.pi * 0.25 * time)
carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

# Compute and plot the spectrogram.

f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#%%------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  # For reading the .wav file.

# fs: sampling frequency.
# signal: the numpy 2D array where the data of the wav file is written.
[fs, signal] = scipy.io.wavfile.read(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav')

length = len(signal)  # The length of the wav file. The number of samples, not the length in time.

window_hop_length = 0.01  # 10ms change here.
overlap = int(fs * window_hop_length)
print('overlap =', overlap)

window_size = 0.025  # 25 ms change here.
framesize = int(window_size * fs)
print('framesize =', framesize)

number_of_frames = int(length / overlap)
nfft_length = framesize  # Length of DFT.
print('number of frames =', number_of_frames)

# Declare a 2D matrix, with rows equal to the number of frames, and columns equal to the framesize or the length of each DFT.
frames = np.ndarray((number_of_frames, framesize))

for k in range(0, number_of_frames):
	for i in range(0, framesize):
		if (k * overlap + i) < length:
			#frames[k][i] = signal[k * overlap + i]
			frames[k][i] = signal[k * overlap + i, 1]
		else:
			frames[k][i] = 0

fft_matrix = np.ndarray((number_of_frames, framesize))  # Declare another 2D matrix to store the DFT of each windowed frame.
abs_fft_matrix = np.ndarray((number_of_frames, framesize))  # Declare another 2D matrix to store the power spectrum.
for k in range(0, number_of_frames):
	fft_matrix[k] = np.fft.fft(frames[k])  # Compute the DFT.
	abs_fft_matrix[k] = abs(fft_matrix[k]) * abs(fft_matrix[k]) / max(abs(fft_matrix[k]))  # Compute the power spectrum.

# Plot the power spectrum obtained above.
t = range(len(abs_fft_matrix))
plt.plot(t, abs_fft_matrix)
plt.ylabel('frequency')
plt.xlabel('time')
plt.show()
