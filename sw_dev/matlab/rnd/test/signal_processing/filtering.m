frequency1 = 5.0; amplitude1 = 2.0;
frequency2 = 20.0; amplitude2 = 1.0;
frequency3 = 35.0; amplitude3 = 4.0;
frequency4 = 60.0; amplitude4 = 3.0;

samplingRate = 1000.0;  % [Hz].

startTime = 0.0;  % [sec].
endTime = 2.0;  % [sec].

numSignal = ceil((endTime - startTime) * samplingRate);
x = zeros([numSignal + 1, 1]);
idx = 1;
for t = startTime:1/samplingRate:endTime
	x(idx) = amplitude1 * sin(2.0 * pi * frequency1 * t) + ...
		amplitude2 * sin(2.0 * pi * frequency2 * t) + ...
		amplitude3 * sin(2.0 * pi * frequency3 * t) + ...
		amplitude4 * sin(2.0 * pi * frequency4 * t);
	idx = idx + 1;
end;

%-----------------------------------------------------------
% Butterworth low-pass filter.

Fc = 1000;  % Cut-off frequency [Hz].
Fs = 8192;  % Sampling rate [Hz].
order = 4;  % Filter order.

% Filter coefficients.
[b, a] = butter(order, 2 * Fc / Fs);  % Low-pass filter. [0:pi] maps to [0:1] here.
%[sos, g] = tf2sos(b, a);

% Apply a filter to data.
y = filter(b, a, x);

%-----------------------------------------------------------
% Butterworth band-pass filter.

Fc1 = 10;  % Cut-off frequency [Hz].
Fc2 = 40;  % Cut-off frequency [Hz].
Fs = 1000;  % Sampling rate [Hz].
order = 4;  % Filter order.

% Filter coefficients.
[b, a] = butter(order, [2 * Fc1 / Fs, 2 * Fc2 / Fs]);  % Band-pass filter. [0:pi] maps to [0:1] here.
%[sos, g] = tf2sos(b, a);

% Apply a filter to data.
y = filter(b, a, x);
