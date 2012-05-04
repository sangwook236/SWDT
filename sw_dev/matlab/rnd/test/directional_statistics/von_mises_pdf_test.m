%addpath D:\work_center\sw_dev\matlab\rnd\src\directional_statistics\CircStat2010b

mean_dir = 0.0;

kappa = 30.0;
%sigma = besseli(1, kappa) / besseli(0, kappa);  % as kappa -> inf
sigma = 1 / sqrt(kappa);
%scalefactor = 1;
scalefactor = 1.05 * von_mises_pdf(pi, mean_dir, kappa) / normpdf(pi, mean_dir, sigma);

%kappa = 1.0;
%sigma = 1.55;
%scalefactor = 1.05 * von_mises_pdf(pi, mean_dir, kappa) / normpdf(pi, mean_dir, sigma);

figure;
dir1 = linspace(-pi, pi, 10000);
vonmisespdf1 = von_mises_pdf(dir1, mean_dir, kappa);
normpdf1 = normpdf(dir1, mean_dir, sigma) * scalefactor;
plot(dir1, vonmisespdf1, 'r', dir1, normpdf1, 'b');
 
figure;
dir2 = linspace(0, 2 * pi, 10000);
vonmisespdf2 = von_mises_pdf(dir2, mean_dir, kappa);
normpdf2 = normpdf(dir2, mean_dir, sigma) * scalefactor;
plot(dir2, vonmisespdf2, 'r', dir2, normpdf2, 'b');
