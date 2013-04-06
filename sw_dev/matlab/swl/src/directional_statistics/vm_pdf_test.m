%addpath('D:\work_center\sw_dev\matlab\rnd\src\directional_statistics\circstat\CircStat2012a');

mean_dir = 0.0;

kappa = 30.0;
%sigma = besseli(1, kappa) / besseli(0, kappa);  % as kappa -> inf
sigma = 1 / sqrt(kappa);
%scalefactor = 1;
scalefactor = 1.05 * vm_pdf(pi, mean_dir, kappa) / normpdf(pi, mean_dir, sigma);

%kappa = 1.0;
%sigma = 1.55;
%scalefactor = 1.05 * vm_pdf(pi, mean_dir, kappa) / normpdf(pi, mean_dir, sigma);

figure;
dir1 = linspace(-pi, pi, 10000);
vmpdf1 = vm_pdf(dir1, mean_dir, kappa);
normpdf1 = normpdf(dir1, mean_dir, sigma) * scalefactor;
plot(dir1, vmpdf1, 'r', dir1, normpdf1, 'b');
 
figure;
dir2 = linspace(0, 2 * pi, 10000);
vmpdf2 = vm_pdf(dir2, mean_dir, kappa);
normpdf2 = normpdf(dir2, mean_dir, sigma) * scalefactor;
plot(dir2, vmpdf2, 'r', dir2, normpdf2, 'b');
