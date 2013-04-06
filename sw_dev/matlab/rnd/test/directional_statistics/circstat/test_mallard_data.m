dir_data = [
10	40
30	22
50	20
70	9
90	6
110	3
130	3
150	1
170	6
190	3
210	11
230	22
250	24
270	58
290	136
310	138
330	143
350	69
];

alpha = dir_data(:,1);
w = dir_data(:,2);
N = length(alpha);

% spacing of bin centers for binned data
bin_spacing = 20;  % [deg]
bin_spacing = circ_ang2rad(bin_spacing);  % [rad]

alpha = circ_ang2rad(alpha);

% mean direction
mu = circ_mean(alpha, w);
circ_rad2ang(mu)

% circular variance
var = circ_var(alpha, w);
var

% median direction
med = circ_median(alpha);
circ_rad2ang(med)

%
stats = circ_stats(alpha, w);
stats

%
[mp rho_p mu_p] = circ_moment(alpha, w, 1)
