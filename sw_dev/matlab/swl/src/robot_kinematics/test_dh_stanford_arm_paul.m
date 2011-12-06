%
% using Paul's notation
%	Fu's book pp. 38
%

%------------------------------------------------------------------------------
d1 = 150.0;
d2 = 100.0;
d3 = 300.0;
d6 = 50.0;

%------------------------------------------------------------------------------
T0 = eye(4);
T1 = [
	0	0	1	0
	-1	0	0	0
	0	-1	0	d1
	0	0	0	1
];
T2 = [
	0	1	0	d2
	0	0	1	0
	1	0	0	d1
	0	0	0	1
];
T3 = [
	-1	0	0	d2
	0	0	1	d3
	0	1	0	d1
	0	0	0	1
];
T4 = [
	-1	0	0	d2
	0	-1	0	d3
	0	0	1	d1
	0	0	0	1
];
T5 = [
	-1	0	0	d2
	0	0	1	d3
	0	1	0	d1
	0	0	0	1
];
T6 = [
	-1	0	0	d2
	0	0	1	d3 + d6
	0	1	0	d1
	0	0	0	1
];

a = zeros(6, 1);
alpha = zeros(6, 1);
d = zeros(6, 1);
theta = zeros(6, 1);

idx = 1;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T0, T1, 'paul');
idx = 2;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T1, T2, 'paul');
idx = 3;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T2, T3, 'paul');
idx = 4;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T3, T4, 'paul');
idx = 5;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T4, T5, 'paul');
idx = 6;
[ a(idx), alpha(idx), d(idx), theta(idx) ] = calc_dh(T5, T6, 'paul');

dh_param1 = [ a, alpha, d, theta ];

%------------------------------------------------------------------------------
T = cell(1, 7);
T{1} = T0;
T{2} = T1;
T{3} = T2;
T{4} = T3;
T{5} = T4;
T{6} = T5;
T{7} = T6;

dh_param2 = calc_dh_param(T, 'paul');
