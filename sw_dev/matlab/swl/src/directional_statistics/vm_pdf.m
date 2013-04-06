function prob = vm_pdf(dir, mean_dir, kappa)

% von Mises distribution (1-dimensional)
%
% dir: a direction angle, [rad].
% mean_dir: a mean direction angle, [rad].
% kappa: a concentration parameter, kappa >= 0.

prob = exp(kappa * cos(dir - mean_dir)) / (2 * pi * besseli(0, kappa));
