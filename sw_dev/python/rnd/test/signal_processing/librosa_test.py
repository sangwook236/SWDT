#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# REF [site] >> https://librosa.github.io/librosa/generated/librosa.filters.mel.html
def mel_filter_bank_test():
	mel_fb = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128)
	#mel_fb = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=128, fmax=8000)  # Clip the maximum frequency to 8KHz.

	#--------------------
	plt.figure()
	librosa.display.specshow(mel_fb, x_axis='linear')
	plt.ylabel('Mel filter')
	plt.title('Mel Filter Bank')
	plt.colorbar()
	plt.tight_layout()
	plt.show()

# REF [site] >> https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
def melspectrogram_test():
	y, sr = librosa.load(librosa.util.example_audio_file())

	if False:
		S = librosa.feature.melspectrogram(y=y, sr=sr)
	elif False:
		D = np.abs(librosa.stft(y))**2
		S = librosa.feature.melspectrogram(S=D, sr=sr)
	else:
		# Passing through arguments to the Mel filters.
		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
		#S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=128, fmax=8000)

	#--------------------
	plt.figure(figsize=(10, 4))
	S_dB = librosa.power_to_db(S, ref=np.max)
	librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel-frequency Spectrogram')
	plt.tight_layout()
	plt.show()

def main():
	mel_filter_bank_test()
	melspectrogram_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
