Fs = 1000;  % sampling frequency, [Hz].
Ts = 1 / Fs;  % sampling time, [sec].
Ls = 1000;  % length of signal.
%Ls = 10000;  % length of signal.
t = (0:(Ls - 1)) * Ts;  % time vector.

% sum of a 50 Hz sinusoid and a 120 Hz sinusoid.
x = 0.7 * sin(2*pi*50*t) + sin(2*pi*120*t); 
y = x + 2 * randn(size(t));  % sinusoids plus noise.

figure;
plot(t, y)
%plot(t(1:50), y(1:50))
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('time (seconds)')

% FFT ---------------------------------------------------------------

fft_len = 2^nextpow2(Ls);  % next power of 2 from length of y.
%fft_dat = fft(y, fft_len);
fft_dat = fft(y, fft_len) / Ls;
% the first component of FFT, FFT(1), is simply the sum of the data,
% and can be removed.
%fft_dat(1) = [];

figure;
plot(fft_dat, 'ro')
title('Fourier Coefficients in the Complex Plane');
xlabel('Real Axis');
ylabel('Imaginary Axis');

% The frequency resolution is dependent on the relationship between the FFT length and the sampling rate of the input signal.
%
% If the sampling rate of the signal is 10kHz and we collect 8192 samples for the FFT then we will have:
%	8192 / 2 = 4096 FFT lines
% Since, via nyquist, our signal contains content up to 5kHz our line resolution is:
%	5000Hz / 4096 lines = 1.22 Hz/line
%
% This is may be the easier way to explain it conceptually but simplified, your line resolution is just:
%	Fs / N
% where Fs is the input signal's sampling rate and N is the number of FFT points used.
%
% We can see from the above that to get smaller FFT lines we can either run a longer FFT or decrease our sampling rate.

fft_line_num = fft_len / 2;
nyquist = 1 / 2;
fft_line_resolution = Fs * nyquist / fft_line_num;
%fft_line_resolution = Fs / fft_len;

freq = Fs * nyquist * linspace(0, 1, fft_line_num + 1);

% amplitude spectrum.
%amplitude_spectrum = abs(fft_dat(1:(fft_line_num + 1)));  % two-sided amplitude spectrum.
amplitude_spectrum = 2 * abs(fft_dat(1:(fft_line_num + 1)));  % single-sided amplitude spectrum.
amplitude_spectrum(1) = amplitude_spectrum(1) * 2;

figure;
plot(freq, amplitude_spectrum) 
title('Single-Sided Amplitude Spectrum of y(t)')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

% The complex magnitude squared of FFT is called the power,
% and a plot of power versus frequency is a "periodogram".
%power = abs(fft_dat(1:(fft_line_num + 1))).^2;  % two-sided power spectrum.
power = 2 * abs(fft_dat(1:(fft_line_num + 1))).^2;  % single-sided power spectrum.
power(1) = power(1) * 2;

figure;
plot(freq, power)
xlabel('Frequency')
ylabel('Power');
title('Periodogram')

figure;
plot(freq(1:200), power(1:200))
xlabel('Frequency')
ylabel('Power');
title('Periodogram')

% We can fix the cycle length a little more precisely by picking out the strongest frequency.
% The red dot locates this point.
hold on;
index = find(power == max(power));
max_freq_str = num2str(freq(index));
max_power_str = num2str(power(index));
plot(freq(index), power(index), 'r.', 'MarkerSize', 25);
text(freq(index) + 5, power(index), [ 'Freq=', max_freq_str, ', Power=', max_power_str ]);
hold off;

period = 1 ./ freq;

figure;
plot(period, power);
axis([0 0.1 0 max(power) * 1.1]);
xlabel('Period');
ylabel('Power');

% We can fix the cycle length a little more precisely by picking out the strongest frequency.
% The red dot locates this point.
hold on;
index = find(power == max(power));
max_period_str = num2str(period(index));
max_power_str = num2str(power(index));
plot(period(index), power(index), 'r.', 'MarkerSize', 25);
text(period(index) + 0.005, power(index), [ 'Period=', max_period_str, ', Power=', max_power_str ]);
hold off;
