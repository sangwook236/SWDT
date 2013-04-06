dir_data = [
15	9
45	23
75	24
105	13
135	2
165	5
];

alpha = dir_data(:,1);
w = dir_data(:,2);
N = length(alpha);

% the range of angles, [0, pi]
l = 2;

% spacing of bin centers for binned data
bin_spacing = 30;  % [deg]
bin_spacing = circ_ang2rad(bin_spacing);  % [rad]

% double the angles
alpha = circ_ang2rad(alpha * l);

% mean direction
mu = circ_mean(alpha, w);
mu = mu / l;
circ_rad2ang(mu)

% circular variance
var = circ_var(alpha, w);
var = 1 - (1 - var)^(1/l^2);
var

% median direction
med = circ_median(alpha);
circ_rad2ang(med)

%
stats = circ_stats(alpha, w);
stats
