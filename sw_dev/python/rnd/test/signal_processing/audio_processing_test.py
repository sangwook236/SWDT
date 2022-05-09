#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np

def sounddevice_test():
	import sounddevice as sd

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/work/dataset'
	else:
		data_home_dir_path = 'D:/work/dataset'
	data_dir_path = data_home_dir_path + '/failure_analysis/defect/knock_sound/500-1500Hz'

	fs = 44100
	data = np.random.uniform(-1, 1, fs)
	sd.play(data, fs)

	sd.default.dtype

	sd.query_devices()

	sd.default.samplerate = 44100
	# A single value sets both input and output at the same time.
	#sd.default.device = 'digital output'
	#sd.default.device = 7
	# Different values for input and output.
	sd.default.channels = 5, 7

	sd.play(data)

	sd.default.reset()

#--------------------------------------------------------------------

def scipy_test():
	import scipy.io.wavfile  # For reading the .wav file.
	import sounddevice as sd

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/work/dataset'
	else:
		data_home_dir_path = 'D:/work/dataset'
	data_dir_path = data_home_dir_path + '/failure_analysis/defect/knock_sound/500-1500Hz'

	# fs: sampling frequency.
	# signal: the numpy 2D array where the data of the wav file is written.
	[fs, signal] = scipy.io.wavfile.read(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav')

	sd.play(signal)

	length = len(signal)  # The length of the wav file. The number of samples, not the length in time.

	window_hop_length = 0.01  # 10ms change here.
	overlap = int(fs * window_hop_length)
	print('overlap = {}.'.format(overlap))

	window_size = 0.025  # 25 ms change here.
	framesize = int(window_size * fs)
	print('framesize = {}.'.format(framesize))

	number_of_frames = int(length / overlap)
	nfft_length = framesize  # Length of DFT.
	print('number of frames = {}.'.format(number_of_frames))

	# Declare a 2D matrix, with rows equal to the number of frames, and columns equal to the framesize or the length of each DFT.
	frames = np.ndarray((number_of_frames, framesize))

#--------------------------------------------------------------------

def pyaudio_test():
	import wave, pyaudio

	if 'posix' == os.name:
		data_home_dir_path = '/home/sangwook/work/dataset'
	else:
		data_home_dir_path = 'D:/work/dataset'
	data_dir_path = data_home_dir_path + '/failure_analysis/defect/knock_sound/500-1500Hz'

	# Open a wave file.
	wavefile = wave.open(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav', 'rb')

	# Create PyAudio.
	p = pyaudio.PyAudio()

	# Open a stream. 
	stream = p.open(
		format=p.get_format_from_width(wavefile.getsampwidth()),
		channels=wavefile.getnchannels(),
		rate=wavefile.getframerate(),
		output=True
	)

	# Define a stream chunk.
	chunk = 1024

	# Read data.
	data = wavefile.readframes(chunk)

	# Play the stream.
	while data:
		stream.write(data)
		data = wavefile.readframes(chunk)

	# Stop the stream.
	stream.stop_stream()
	stream.close()

	# Close PyAudio.
	p.terminate()
	wavefile.close()

def main():
	sounddevice_test()
	scipy_test()
	pyaudio_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
