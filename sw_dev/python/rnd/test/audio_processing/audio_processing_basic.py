#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/sangwook/my_dataset"
#dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
dataset_home_dir_path = "D:/dataset"

data_dir_path = dataset_home_dir_path + "/failure_analysis/defect/motor_20170621/0_original/500-1500Hz"

#%%------------------------------------------------------------------

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

#%%------------------------------------------------------------------

import numpy as np
import scipy.io.wavfile  # For reading the .wav file.
import sounddevice as sd

# fs: sampling frequency.
# signal: the numpy 2D array where the data of the wav file is written.
[fs, signal] = scipy.io.wavfile.read(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav')

sd.play(signal)

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

#%%------------------------------------------------------------------

import wave, pyaudio

# Open a wave file.
wavefile = wave.open(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav', 'rb')

# Create PyAudio.
p = pyaudio.PyAudio()

# Open a stream. 
stream = p.open(format = p.get_format_from_width(wavefile.getsampwidth()),
                channels = wavefile.getnchannels(),
                rate = wavefile.getframerate(),
                output = True)

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

#%%------------------------------------------------------------------

import Sound

s = Sound()
s.read(data_dir_path + '/KMHFF41CBBA036937_2000RPM ( 0.00- 5.73 s).wav')
s.play()
