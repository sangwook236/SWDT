function prob = von_mises_pdf(dir, mean_dir, kappa)

% dir: direction, [rad]

prob = exp(kappa * cos(dir - mean_dir)) / (2 * pi * besseli(0, kappa));
