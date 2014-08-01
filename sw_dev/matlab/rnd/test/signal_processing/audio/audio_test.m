%--------------------------------------------------------------------
% noise
%	White noise has an equal amount of energy on every frequency.
%	In music, there is often band-limited noise present.

[noise_wav, noise_Fs] = audioread('noise.wav');
figure;
subplot(2, 1, 1);
noise_time = ((1:length(noise_wav)) - 1) * (1 / noise_Fs);
plot(noise_time, noise_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(noise_wav);
spectrogram(noise_wav, 256, [], 256, noise_Fs);
colorbar;

%--------------------------------------------------------------------
% speech
%	The speech sample shown is the finnish word "seitseman".

[speech_wav, speech_Fs] = audioread('seiska.wav');
figure;
subplot(2, 1, 1);
speech_time = ((1:length(speech_wav)) - 1) * (1 / speech_Fs);
plot(speech_time, speech_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(speech_wav);
spectrogram(speech_wav, 256, [], 256, speech_Fs);
colorbar;

%--------------------------------------------------------------------
% piano
%	The piano sample shown is the middle C, whose fundamental frequency is 261 Hz.
%	The piano sample is an example of a harmonic sound; this means that the sound consists of sine waves which are integer multiples of the fundamental frequency. (Actually the piano is not perfectly harmonic.)

[piano_wav, piano_Fs] = audioread('pia60.wav');
figure;
subplot(2, 1, 1);
piano_time = ((1:length(piano_wav)) - 1) * (1 / piano_Fs);
plot(piano_time, piano_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(piano_wav);
spectrogram(piano_wav, 256, [], 256, piano_Fs);
colorbar;

%--------------------------------------------------------------------
% snare drum
%	The snare drum sample doesn't have a fundamental frequency nor does it have overtones.

[drum_wav, drum_Fs] = audioread('snareHit.wav');
figure;
subplot(2, 1, 1);
drum_time = ((1:length(drum_wav)) - 1) * (1 / drum_Fs);
plot(drum_time, drum_wav);
axis tight;
grid on;
subplot(2, 1, 2);
%specgram(drum_wav);
spectrogram(drum_wav, 256, [], 256, drum_Fs);
colorbar;

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

[drum_wav, drum_Fs] = audioread('snareHit.wav');
figure;
subplot(2, 1, 1);
drum_time = ((1:length(drum_wav)) - 1) * (1 / drum_Fs);
plot(drum_time, drum_wav);
title('drum signal');
subplot(2, 1, 2);
drum_fft_len = 2^nextpow2(length(drum_wav));  % Next power of 2 from length of y.
%drum_fft_line_resolution = drum_Fs / drum_fft_len;
drum_freq = drum_Fs / 2 * linspace(0, 1, drum_fft_len / 2 + 1);
drum_fft = fft(drum_wav, drum_fft_len) / length(drum_wav);
drum_fft_amplitude = 2 * abs(drum_fft(1:(drum_fft_len / 2 + 1)));  % single-sided amplitude spectrum.
drum_fft_amplitude(1) = drum_fft_amplitude(1) * 2;
plot(drum_freq, drum_fft_amplitude);
title('drum single-sided amplitude spectrum');

[drum_pks, drum_locs] = findpeaks(drum_fft_amplitude, 'MinPeakHeight', 0.007);
drum_peak_freq = drum_freq(drum_locs(1))
%drum_freq(drum_locs)

%
drum_u = drum_wav((0.2 * drum_Fs):(0.2 * drum_Fs + 511)) .* hanning(512);
drum_U = fft(drum_u);
drum_f = (0:256) / 256 * drum_Fs / 2;

figure;
plot(drum_f, 20 * log10(abs(drum_U(1:257))))
axis tight;
grid on;
xlabel('frequency [Hz]');
ylabel('amplitude [dB]');

%--------------------------------------------------------------------
% piano - windowing

%[piano_wav, piano_Fs] = audioread('pia60.wav');
%[piano_wav, piano_Fs] = audioread('pia60.wav', [5000, 6000]);

figure;
subplot(2, 1, 1);
piano_time = ((1:length(piano_wav)) - 1) * (1 / piano_Fs);
plot(piano_time, piano_wav);
title('piano signal');
subplot(2, 1, 2);
piano_fft_len = 2^nextpow2(length(piano_wav));  % Next power of 2 from length of y.
%piano_fft_line_resolution = piano_Fs / piano_fft_len;
piano_freq = piano_Fs / 2 * linspace(0, 1, piano_fft_len / 2 + 1);
piano_fft = fft(piano_wav, piano_fft_len) / length(piano_wav);
piano_fft_amplitude = 2 * abs(piano_fft(1:(piano_fft_len / 2 + 1)));  % single-sided amplitude spectrum.
piano_fft_amplitude(1) = piano_fft_amplitude(1) * 2;
plot(piano_freq, piano_fft_amplitude);
title('piano single-sided amplitude spectrum');

[piano_pks, piano_locs] = findpeaks(piano_fft_amplitude, 'MinPeakHeight', 0.005);
piano_peak_freq = piano_freq(piano_locs(1))
%piano_freq(piano_locs)

%
piano_S1 = fft(piano_wav(1:512));
piano_S2 = fft(piano_wav(1:512) .* hanning(512));
piano_f = (0:256) / 256 * piano_Fs / 2;

figure;
plot(piano_f, 20 * log10(abs(piano_S1(1:257))), 'r')
hold on;
plot(piano_f, 20 * log10(abs(piano_S2(1:257))), 'b')
hold off;
axis tight;
grid on;
xlabel('frequency [Hz]')
ylabel('amplitude [dB]')
