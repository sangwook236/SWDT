% [ref] http://www.cs.tut.fi/sgn/arg/intro/basics.html

%--------------------------------------------------------------------
% noise
%	White noise has an equal amount of energy on every frequency.
%	In music, there is often band-limited noise present.

[noise_wav, noise_Fs] = audioread('noise.wav');

noise_Ts = 1 / noise_Fs;  % sampling time, [sec].
noise_wav_len = length(noise_wav);  % length of signal.

figure;
subplot(2, 1, 1);
noise_time = ((1:noise_wav_len) - 1) * noise_Ts;
plot(noise_time, noise_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(noise_wav);
spectrogram(noise_wav, 256, [], 256, noise_Fs);
%spectrogram(noise_wav, 256, [], 256, noise_Fs, 'yaxis');
colorbar;

%--------------------------------------------------------------------
% speech
%	The speech sample shown is the finnish word "seitseman".

[speech_wav, speech_Fs] = audioread('seiska.wav');

speech_Ts = 1 / speech_Fs;  % sampling time, [sec].
speech_wav_len = length(speech_wav);  % length of signal.

figure;
subplot(2, 1, 1);
speech_time = ((1:speech_wav_len) - 1) * speech_Ts;
plot(speech_time, speech_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(speech_wav);
spectrogram(speech_wav, 256, [], 256, speech_Fs);
%spectrogram(speech_wav, 256, [], 256, speech_Fs, 'yaxis');
colorbar;

%--------------------------------------------------------------------
% piano
%	The piano sample shown is the middle C, whose fundamental frequency is 261 Hz.
%	The piano sample is an example of a harmonic sound; this means that the sound consists of sine waves which are integer multiples of the fundamental frequency. (Actually the piano is not perfectly harmonic.)

[piano_wav, piano_Fs] = audioread('pia60.wav');

piano_Ts = 1 / piano_Fs;  % sampling time, [sec].
piano_wav_len = length(piano_wav);  % length of signal.

figure;
subplot(2, 1, 1);
piano_time = ((1:piano_wav_len) - 1) * piano_Ts;
plot(piano_time, piano_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(piano_wav);
spectrogram(piano_wav, 256, [], 256, piano_Fs);
%spectrogram(piano_wav, 256, [], 256, piano_Fs, 'yaxis');
colorbar;

figure;
piano_wav_resampled = resample(piano_wav, 1, 3);
%specgram(piano_wav_resampled, 512, piano_Fs / 3);
spectrogram(piano_wav, 2^12, [], 2^12, piano_Fs / 3);
colorbar
ax = axis; axis([ax(1) 1500 ax(3) ax(4)]);

%--------------------------------------------------------------------
% snare drum
%	The snare drum sample doesn't have a fundamental frequency nor does it have overtones.

[drum_wav, drum_Fs] = audioread('snareHit.wav');

drum_Ts = 1 / drum_Fs;  % sampling time, [sec].
drum_wav_len = length(drum_wav);  % length of signal.

figure;
subplot(2, 1, 1);
drum_time = ((1:drum_wav_len) - 1) * drum_Ts;
plot(drum_time, drum_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(drum_wav);
spectrogram(drum_wav, 256, [], 256, drum_Fs);
%spectrogram(drum_wav, 256, [], 256, drum_Fs, 'yaxis');
colorbar;

figure;
drum_wav_resampled = resample(drum_wav, 1, 3);
%specgram(drum_wav_resampled, 512, drum_Fs / 3);
spectrogram(drum_wav, 2^12, [], 2^12, drum_Fs / 3);
colorbar
ax = axis; axis([ax(1) 1000 ax(3) ax(4)]);

%--------------------------------------------------------------------
% snare drum - windowing
%	The Fourier transform decomposes a signal into a sum of stationary sinusoids.
%	Therefore, when a whole regular sound signal is transformed, the changes in frequency content cannot be observed.
%	Therefore short-time windowed FFT is usually used to observe the instantaneous frequency content.
%
%	Short-time signal processing is practically always done using windowing.
%	In short-time signal processing, signals are cut into small pieces called frames, which are processed one at a time.
%	Frames are windowed with a window function in order to improve the frequency-domain representation.
%	What windowing essentially means is multiplying the signal frame with the window function point-by-point.

%[drum_wav, drum_Fs] = audioread('snareHit.wav');

%drum_Ts = 1 / drum_Fs;  % sampling time, [sec].
%drum_wav_len = length(drum_wav);  % length of signal.

% two-sided amplitude spectrum.
drum_fft_len = 2^nextpow2(drum_wav_len);  % Next power of 2 from length of y.
%drum_fft_line_resolution = drum_Fs / drum_fft_len;
%drum_fft = fft(drum_wav, drum_fft_len);
drum_fft = fft(drum_wav, drum_fft_len) / drum_wav_len;

% single-sided amplitude spectrum.
drum_fft_amplitude = 2 * abs(drum_fft(1:(drum_fft_len / 2 + 1)));
drum_fft_amplitude(1) = drum_fft_amplitude(1) * 2;

drum_time = ((1:drum_wav_len) - 1) * drum_Ts;
drum_freq = drum_Fs / 2 * linspace(0, 1, drum_fft_len / 2 + 1);

figure;
subplot(2, 1, 1);
plot(drum_time, drum_wav);
title('drum signal');
subplot(2, 1, 2);
plot(drum_freq, drum_fft_amplitude);
title('drum single-sided amplitude spectrum');

[drum_pks, drum_locs] = findpeaks(drum_fft_amplitude, 'MinPeakHeight', 0.007);
drum_peak_freq = drum_freq(drum_locs(1))
%drum_freq(drum_locs)

% windowing
drum_win_len = 512;  % the length of window.
drum_win_overlap = drum_win_len / 2;  % The number of samples each segment of X overlaps.
drum_freq_windowed = linspace(0, 1, drum_win_len / 2 + 1) * drum_Fs / 2;

% short-time Fourier transform (STFT).
%	A wide window gives better frequency resolution but poor time resolution.
%	A narrower window gives good time resolution but poor frequency resolution.

%drum_wav_windowed = drum_wav((0.2 * drum_Fs):(0.2 * drum_Fs + 511)) .* hanning(drum_win_len);
drum_wav_start = 0.2 * drum_Fs;
drum_wav_end = drum_wav_start + drum_win_len - 1;
drum_wav_windowed = drum_wav(drum_wav_start:drum_wav_end) .* hanning(drum_win_len);
%drum_wav_windowed = drum_wav(drum_wav_start:drum_wav_end) .* rectwin(drum_win_len);  % for analyzing transients.
drum_stft = fft(drum_wav_windowed);

figure;
plot(drum_freq_windowed, 20 * log10(abs(drum_stft(1:(drum_win_len / 2 + 1)))))
axis tight;
grid on;
xlabel('frequency [Hz]');
ylabel('amplitude [dB]');

%--------------------------------------------------------------------
% piano - windowing

%[piano_wav, piano_Fs] = audioread('pia60.wav');
%[piano_wav, piano_Fs] = audioread('pia60.wav', [5000, 6000]);

%piano_Ts = 1 / piano_Fs;  % sampling time, [sec].
%piano_wav_len = length(piano_wav);  % length of signal.

% two-sided amplitude spectrum.
piano_fft_len = 2^nextpow2(piano_wav_len);  % Next power of 2 from length of y.
%piano_fft_line_resolution = piano_Fs / piano_fft_len;
%piano_fft = fft(piano_wav, piano_fft_len);
piano_fft = fft(piano_wav, piano_fft_len) / piano_wav_len;

% single-sided amplitude spectrum.
piano_fft_amplitude = 2 * abs(piano_fft(1:(piano_fft_len / 2 + 1)));
piano_fft_amplitude(1) = piano_fft_amplitude(1) * 2;

piano_time = ((1:piano_wav_len) - 1) * piano_Ts;
piano_freq = piano_Fs / 2 * linspace(0, 1, piano_fft_len / 2 + 1);

figure;
subplot(2, 1, 1);
plot(piano_time, piano_wav);
title('piano signal');
subplot(2, 1, 2);
plot(piano_freq, piano_fft_amplitude);
title('piano single-sided amplitude spectrum');

[piano_pks, piano_locs] = findpeaks(piano_fft_amplitude, 'MinPeakHeight', 0.005);
piano_peak_freq = piano_freq(piano_locs(1))
%piano_freq(piano_locs)

% windowing
piano_win_len = 512;  % the length of window.
piano_win_overlap = piano_win_len / 2;  % The number of samples each segment of X overlaps.
piano_freq_windowed = linspace(0, 1, piano_win_len / 2 + 1) * piano_Fs / 2;

% short-time Fourier transform (STFT).
%	A wide window gives better frequency resolution but poor time resolution.
%	A narrower window gives good time resolution but poor frequency resolution.

piano_stft1 = fft(piano_wav(1:piano_win_len) .* rectwin(piano_win_len));  % using uniform or rectangular window. for analyzing transients.
piano_stft2 = fft(piano_wav(1:piano_win_len) .* hanning(piano_win_len));  % using Hann window.

figure;
plot(piano_freq_windowed, 20 * log10(abs(piano_stft1(1:(piano_win_len / 2 + 1)))), 'r')
hold on;
plot(piano_freq_windowed, 20 * log10(abs(piano_stft2(1:(piano_win_len / 2 + 1)))), 'b')
hold off;
axis tight;
grid on;
xlabel('frequency [Hz]')
ylabel('amplitude [dB]')
