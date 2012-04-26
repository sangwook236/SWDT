%addpath D:\work_center\sw_dev\matlab\rnd\src\directional_statistics\CircStat2010b

Ndata = 180 * 181;

dir_data = zeros(Ndata,1);
kk = 1;
for ii = 1:360
	ll = mod(ii, 180);
	for jj = 1:ll
		dir_data(kk) = ii - 1;
		kk = kk + 1;
	end;
end;

%alpha = dir_data(:,1);
%weighting = dir_data(:,2);
alpha = dir_data;

alpha = circ_ang2rad(alpha);

%--------------------------------------------------------------------
figure;
subplot(2,2,1);
circ_plot(alpha, 'pretty', 'ro', true, 'linewidth', 2, 'color', 'r');
title('pretty plot style');
subplot(2,2,2);
binCount = 360;
circ_plot(alpha, 'hist', [], binCount, true, true, 'linewidth', 2, 'color', 'r');
title('hist plot style');
subplot(2,2,3);
%circ_plot(alpha, 'density', 'linewidth', 2, 'color', 'r');
%title('density plot style');
subplot(2,2,4);
circ_plot(alpha, [], 's');
title('non-fancy plot style');

%--------------------------------------------------------------------
% mean direction
mu = circ_mean(alpha);
circ_rad2ang(mu)

% circular variance
var = circ_var(alpha);
var

% median direction
med = circ_median(alpha);
circ_rad2ang(med)

%--------------------------------------------------------------------
stats = circ_stats(alpha);
stats
