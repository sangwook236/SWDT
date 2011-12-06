function [ traj_poses traj_vels traj_accels ] = synthesize_data_using_trigonometric(time, Ts, T, L)

% p(t) = a1 * t + a2 * sin(t) + a3 * sin(2*t) + a4 * sin(3*t) + a5 * cos(t) + a6 * cos(2*t) + a7 * cos(3*t)

%------------------------------------------------------------------------------

% pre-defined coefficient
predefined_coeffs = [ 0 0 0 ];  % [ a2 a3 a4 ]

sin1 = sin(T);
sin2 = sin(2*T);
sin3 = sin(3*T);
cos1 = cos(T);
cos2 = cos(2*T);
cos3 = cos(3*T);
AA = [
	0 0     0       0       1     1       1
	T sin1  sin2    sin3    cos1  cos2    cos3
	1 1     2       3       0     0       0
	1 cos1  2*cos2  3*cos3  -sin1 -2*sin2 -3*sin3
	0 0     0       0       -1    -4      -9
	0 -sin1 -4*sin2 -9*sin3 -cos1 -4*cos2 -9*cos3
];
for ii = 1:3
	BB(:,ii) = [ 0 L(ii) 0 0 0 0 ]';
end;

% coefficients
coeff(:,1) = inv(AA(:,[1 3 4 5 6 7])) * BB(:,1);  % use a2 = 0
coeff(:,2) = inv(AA(:,[1 2 4 5 6 7])) * BB(:,2);  % use a3 = 0
coeff(:,3) = inv(AA(:,[1 2 3 5 6 7])) * BB(:,3);  % use a4 = 0

traj_coeffs(:,1) = [ coeff(1,1) predefined_coeffs(1) coeff(2,1) coeff(3,1) coeff(4,1) coeff(5,1) coeff(6,1) ]';
traj_coeffs(:,2) = [ coeff(1,2) coeff(2,2) predefined_coeffs(2) coeff(3,2) coeff(4,2) coeff(5,2) coeff(6,2) ]';
traj_coeffs(:,3) = [ coeff(1,3) coeff(2,3) coeff(3,3) predefined_coeffs(3) coeff(4,3) coeff(5,3) coeff(6,3) ]';

for ii = 1:3
	[ traj_poses(:,ii) traj_vels(:,ii) traj_accels(:,ii) ] = traj(time, traj_coeffs(:,ii));
end;
