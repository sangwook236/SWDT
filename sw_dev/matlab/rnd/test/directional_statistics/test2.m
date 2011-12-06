dir_data = [
43	1
45	1
52	1
61	1
75	1
88	1
88	1
279	1
357	1
];

alpha = dir_data(:,1);
weighting = dir_data(:,2);
N = length(alpha);

alpha = circ_ang2rad(alpha);

% mean direction
mu = circ_mean(alpha, weighting);
circ_rad2ang(mu)

% circular variance
var = circ_var(alpha, weighting);
var

% median direction
med = circ_median(alpha);
circ_rad2ang(med)

%
stats = circ_stats(alpha, weighting);
stats
