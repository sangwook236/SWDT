function y_n = gps_h(x_n, param)

Ts = param(1);
n = param(2);

% FIXME [modify] >>
r_GPS = [ 0 0 0 ]';
w_k_N = [ 0 0 0 ]';

e0 = x_n(7,:);
e1 = x_n(8,:);
e2 = x_n(9,:);
e3 = x_n(10,:);

Cnb = 2 * [
	0.5-e2^2-e3^2 e1*e2-e0*e3 e1*e3+e0*e2
	e1*e2+e0*e3 0.5-e1^2-e3^2 e2*e3-e0*e1
	e1*e3-e0*e2 e2*e3+e0*e1 0.5-e1^2-e2^2
];

y_n(1:3,:) = x_n(1:3,:) + Cnb * r_GPS;
y_n(4:6,:) = x_n(4:6,:) + Cnb * cross(w_k_N, r_GPS);

if size(x_n,1) > 32
   y_n(1:3,:) = y_n(1:3,:) + x_n(33:35,:);
   y_n(4:6,:) = y_n(4:6,:) + x_n(36:38,:);
end
