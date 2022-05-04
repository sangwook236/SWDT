#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# REF [site] >> https://librosa.org/doc/main/tutorial.html
def beat_tracking_example():
	# Get the file path to an included audio example.
	filename = librosa.example('nutcracker')

	# Load the audio as a waveform 'y' and store the sampling rate as 'sr'.
	y, sr = librosa.load(filename)

	# Run the default beat tracker.
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

	print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

	# Convert the frame indices of beat events into timestamps.
	beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# REF [site] >> https://librosa.org/doc/main/tutorial.html
def feature_extraction_example():
	# Load the example clip.
	y, sr = librosa.load(librosa.ex('nutcracker'))

	# Set the hop length: at 22050 Hz, 512 samples ~= 23ms.
	hop_length = 512

	# Separate harmonics and percussives into two waveforms.
	y_harmonic, y_percussive = librosa.effects.hpss(y)

	# Beat track on the percussive signal.
	tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

	# Compute MFCC features from the raw signal.
	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

	# The first-order differences (delta features).
	mfcc_delta = librosa.feature.delta(mfcc)

	# Stack and synchronize between beat events.
	# This time, we'll use the mean value (default) instead of median.
	beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

	# Compute chroma features from the harmonic signal.
	chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

	# Aggregate chroma features between beat events.
	# We'll use the median value of each feature between beat frames.
	beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

	# Finally, stack all beat-synchronous features together.
	beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

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
	y, sr = librosa.load(librosa.ex('trumpet'))

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

# REF [site] >> https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
def data_augmentation_example():
	y, sr = librosa.load(librosa.example('nutcracker'))
	#y, sr = librosa.load(librosa.example('trumpet'))
	#y, sr = librosa.load(librosa.example('brahms'))
	#y, sr = librosa.load(librosa.example('vibeace', hq=True))

	plt.figure(figsize=(10, 4))
	librosa.display.waveshow(y, sr=sr, x_axis='time')
	plt.title('Original')
	plt.tight_layout()

	#--------------------
	# Inject noise.
	def inject_noise(y, noise_factor):
		noise = np.random.randn(len(y))
		augmented = y + noise_factor * noise
		# Cast back to same data type.
		augmented = augmented.astype(type(y[0]))
		return augmented

	noise_factor = 0.02
	y_augmented = inject_noise(y, noise_factor)

	plt.figure(figsize=(10, 4))
	librosa.display.waveshow(y_augmented, sr=sr, x_axis='time')
	plt.title('Noise Injection')
	plt.tight_layout()

	#--------------------
	# Shift time.
	def shift_time(y, sr, shift_max, shift_direction):
		shift = np.random.randint(sr * shift_max)
		if shift_direction == 'right':
			shift = -shift
		elif shift_direction == 'both':
			direction = np.random.randint(0, 2)
			if direction == 1:
				shift = -shift

		augmented = np.roll(y, shift)
		# Set to silence for heading / tailing.
		if shift > 0:
			augmented[:shift] = 0
		else:
			augmented[shift:] = 0
		return augmented

	shift_max = 10
	shift_direction = 'right'
	y_augmented = shift_time(y, sr, shift_max, shift_direction)

	plt.figure(figsize=(10, 4))
	librosa.display.waveshow(y_augmented, sr=sr, x_axis='time')
	plt.title('Time Shift')
	plt.tight_layout()

	#--------------------
	# Change pitch.
	pitch_factor = 0.2
	y_augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_factor)

	plt.figure(figsize=(10, 4))
	librosa.display.waveshow(y_augmented, sr=sr, x_axis='time')
	plt.title('Pitch Shift')
	plt.tight_layout()

	#--------------------
	# Change speed.
	#	Stretch times series by a fixed rate.
	stretch_factor = 0.8  # If rate < 1, then the signal is slowed down.
	#stretch_factor = 1.2  # If rate > 1, then the signal is sped up.
	y_augmented = librosa.effects.time_stretch(y, rate=stretch_factor)

	plt.figure(figsize=(10, 4))
	librosa.display.waveshow(y_augmented, sr=sr, x_axis='time')
	plt.title('Time Stretch')
	plt.tight_layout()

	plt.show()

def main():
	#beat_tracking_example()
	#feature_extraction_example()

	#--------------------
	#mel_filter_bank_test()
	#melspectrogram_test()

	#--------------------
	data_augmentation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
