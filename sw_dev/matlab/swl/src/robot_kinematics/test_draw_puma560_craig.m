%
% using Craig's notation
%	Craig's book pp. 91
%

%------------------------------------------------------------------------------
a2 = 431.8;
a3 = 20.32;
d3 = 149.09;
d4 = 433.07;

%------------------------------------------------------------------------------
T0 = eye(4);
T1 = eye(4);
T2 = [
	1	0	0	0
	0	0	1	0
	0	-1	0	0
	0	0	0	1
];
T3 = [
	1	0	0	a2
	0	0	1	d3
	0	-1	0	0
	0	0	0	1
];
T4 = [
	1	0	0	a2 + a3
	0	-1	0	d3
	0	0	-1	-d4
	0	0	0	1
];
T5 = [
	1	0	0	a2 + a3
	0	0	1	d3
	0	-1	0	-d4
	0	0	0	1
];
T6 = [
	1	0	0	a2 + a3
	0	-1	0	d3
	0	0	-1	-d4
	0	0	0	1
];

T_init = cell(1, 7);
T_init{1} = T0;
T_init{2} = T1;
T_init{3} = T2;
T_init{4} = T3;
T_init{5} = T4;
T_init{6} = T5;
T_init{7} = T6;

%------------------------------------------------------------------------------
notation = 'craig';

% [ a alpha d theta ]
dh_param = calc_dh_param(T_init, notation);

%------------------------------------------------------------------------------
joint_type = [ 1 1 1 1 1 1 ];
axis_length = 100;

figure;
%qq = [-135:1:135] * pi / 180;
qq = [0];
for ii = 1:length(qq)
	clf;

	q = [ qq(ii) 0 0 0 0 0 ];
	T = calc_robot_pose(dh_param, notation, joint_type, q);
	
	draw_robot_frame(T, axis_length);
end;
