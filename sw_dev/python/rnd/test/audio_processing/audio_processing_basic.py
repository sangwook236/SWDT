#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/sangwook/my_dataset"
#dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
dataset_home_dir_path = "D:/dataset"

data_dir_path = dataset_home_dir_path + "/failure_analysis/defect/motor_20170621/0_original/500-1500Hz"

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

# Play the stream . 
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

