%Xt = sin(linspace(0, 8*pi, 256)) + sin(2 * linspace(0, 8*pi, 256)) + sin(3 * linspace(0, 8*pi, 256)) + sin(7 * linspace(0, 8*pi, 256));
%Xt = sin(linspace(0, 8*pi, 256)) + sin(5 * linspace(0, 8*pi, 256)) + sin(7 * linspace(0, 8*pi, 256)) + sin(13 * linspace(0, 8*pi, 256));
%Xt = cos(linspace(0, 8*pi, 256)) + cos(2 * linspace(0, 8*pi, 256)) + cos(3 * linspace(0, 8*pi, 256)) + cos(7 * linspace(0, 8*pi, 256));
Xt = cos(linspace(0, 8*pi, 256)) + cos(5 * linspace(0, 8*pi, 256)) + cos(7 * linspace(0, 8*pi, 256)) + cos(13 * linspace(0, 8*pi, 256));
N = length(Xt);

% sampling frequency
fs = (256 * 2*pi) / (8*pi - 0);  % [Hz]

% sample autocorrelation sequence
[acs, lag] = sample_acs(Xt, 0, 200);
figure;
plot(lag, acs);

% the power spectral density (PSD) estimate of the sequence x using a periodogram
[Pxx, f] = periodogram(Xt, [], N, fs);

% rescaled periodogram (???)
Pxx_rescaled = Pxx / sum(Pxx);
figure;
plot(f, Pxx_rescaled, '.');

% test of periodicity
alpha = 0.05;
lambda = 0.6;
g_F = 0.1087;
g_0 = lambda * g_F;

ind = find(Pxx_rescaled > g_0);
