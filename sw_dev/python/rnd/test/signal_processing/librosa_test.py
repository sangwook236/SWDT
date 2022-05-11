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

# REF [site] >> https://librosa.org/doc/main/auto_examples/plot_viterbi.html
def viterbi_decoding_example():
	# Problem of silence/non-silence detection.

	y, sr = librosa.load(librosa.ex('trumpet'))

	# Compute the spectrogram magnitude and phase.
	S_full, phase = librosa.magphase(librosa.stft(y))

	# Plot the spectrum.
	fig, ax = plt.subplots()
	img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax)
	fig.colorbar(img, ax=ax)

	# There are periods of silence and non-silence throughout this recording.
	# Plot the root-mean-square (RMS) curve.
	rms = librosa.feature.rms(y=y)[0]
	times = librosa.frames_to_time(np.arange(len(rms)))

	fig, ax = plt.subplots()
	ax.plot(times, rms)
	ax.axhline(0.02, color='r', alpha=0.5)
	ax.set(xlabel='Time', ylabel='RMS')

	# We'll normalize the RMS by its standard deviation to expand the range of the probability vector.
	r_normalized = (rms - 0.02) / np.std(rms)
	p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

	fig, ax = plt.subplots()
	ax.plot(times, p, label='P[V=1|x]')
	ax.axhline(0.5, color='r', alpha=0.5, label='Descision threshold')
	ax.set(xlabel='Time')
	ax.legend()

	# A simple silence detector would classify each frame independently of its neighbors.
	#plt.figure(figsize=(12, 6))
	fig, ax = plt.subplots(nrows=2, sharex=True)
	librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[0])
	ax[0].label_outer()
	ax[1].step(times, p>=0.5, label='Non-silent')
	ax[1].set(ylim=[0, 1.05])
	ax[1].legend()

	# We can do better using the Viterbi algorithm. 
	# We'll assume that a silent frame is equally likely to be followed by silence or non-silence, but that non-silence is slightly more likely to be followed by non-silence.
	# This is accomplished by building a self-loop transition matrix, where transition[i, j] is the probability of moving from state i to state j in the next frame.

	transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
	print(transition)

	# Our p variable only indicates the probability of non-silence, so we need to also compute the probability of silence as its complement.
	full_p = np.vstack([1 - p, p])
	print(full_p)

	# We'll use viterbi_discriminative here, since the inputs are state likelihoods conditional on data (in our case, data is rms).
	states = librosa.sequence.viterbi_discriminative(full_p, transition)

	#sphinx_gallery_thumbnail_number = 5
	fig, ax = plt.subplots(nrows=2, sharex=True)
	librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[0])
	ax[0].label_outer()
	ax[1].step(times, p>=0.5, label='Frame-wise')
	ax[1].step(times, states, linestyle='--', color='orange', label='Viterbi')
	ax[1].set(ylim=[0, 1.05])
	ax[1].legend()

	plt.show()

# REF [site] >> https://librosa.org/doc/main/auto_examples/plot_music_sync.html
def music_synchronization_with_dynamic_time_warping_example():
	# Load audio recordings.
	x_1, fs = librosa.load('./sir_duke_slow.ogg')
	# And a second version, slightly faster.
	x_2, fs = librosa.load('./sir_duke_fast.ogg')

	fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
	librosa.display.waveshow(x_1, sr=fs, ax=ax[0])
	ax[0].set(title='Slower Version $X_1$')
	ax[0].label_outer()

	librosa.display.waveshow(x_2, sr=fs, ax=ax[1])
	ax[1].set(title='Faster Version $X_2$')

	# Extract chroma features.
	hop_length = 1024

	x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, hop_length=hop_length)
	x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, hop_length=hop_length)

	fig, ax = plt.subplots(nrows=2, sharey=True)
	img = librosa.display.specshow(x_1_chroma, x_axis='time', y_axis='chroma', hop_length=hop_length, ax=ax[0])
	ax[0].set(title='Chroma Representation of $X_1$')
	librosa.display.specshow(x_2_chroma, x_axis='time', y_axis='chroma', hop_length=hop_length, ax=ax[1])
	ax[1].set(title='Chroma Representation of $X_2$')
	fig.colorbar(img, ax=ax)

	# Align chroma sequences.
	D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
	wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

	fig, ax = plt.subplots()
	img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=fs, cmap='gray_r', hop_length=hop_length, ax=ax)
	ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
	ax.set(title='Warping Path on Acc. Cost Matrix $D$', xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
	fig.colorbar(img, ax=ax)

	# Alternative visualization in the time domain.
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))
	librosa.display.waveshow(x_2, sr=fs, ax=ax2)  # Plot x_2.
	ax2.set(title='Faster Version $X_2$')
	librosa.display.waveshow(x_1, sr=fs, ax=ax1)  # Plot x_1.
	ax1.set(title='Slower Version $X_1$')
	ax1.label_outer()

	from matplotlib.patches import ConnectionPatch
	n_arrows = 20
	for tp1, tp2 in wp_s[::len(wp_s)//n_arrows]:
		# Create a connection patch between the aligned time points in each subplot.
		con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0), axesA=ax1, axesB=ax2, coordsA='data', coordsB='data', color='r', linestyle='--', alpha=0.5)
		ax2.add_artist(con)

	plt.show()

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

	if True:
		# If a time-series input y, sr is provided, then its magnitude spectrogram S is first computed, and then mapped onto the mel scale by mel_f.dot(S**power).
		S = librosa.feature.melspectrogram(y=y, sr=sr)

		# Passing through arguments to the Mel filters.
		#S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
		#S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=128, fmax=8000)
		#S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, n_mels=128, fmax=8000, htk=True)
	else:
		# If a spectrogram input S is provided, then it is mapped directly onto the mel basis by mel_f.dot(S).
		D = np.abs(librosa.stft(y))**2
		S = librosa.feature.melspectrogram(S=D, sr=sr)

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

	#viterbi_decoding_example()
	music_synchronization_with_dynamic_time_warping_example()

	#--------------------
	#mel_filter_bank_test()
	#melspectrogram_test()

	#--------------------
	#data_augmentation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
