Fc = 1000;  % Cut-off frequency [Hz].
Fs = 8192;  % Sampling rate [Hz].
order = 5;  % Filter order.

% Filter coefficients.
[b, a] = butter(order, 2 * Fc / Fs);  % Low-pass filter. [0:pi] maps to [0:1] here.
%[sos, g] = tf2sos(b, a);

% Apply a filter to data.
y = filter(b, a, x);
