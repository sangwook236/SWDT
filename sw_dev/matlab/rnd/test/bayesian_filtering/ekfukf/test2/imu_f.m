function x_n = imu_f(x, param)

Ts = param(1);
n = param(2);
a_measured = param(3:5);
w_measured = param(6:8);

a_bar = a_measured - x(11:13);
w_bar = w_measured - x(14:16);

e0 = x(7,:);
e1 = x(8,:);
e2 = x(9,:);
e3 = x(10,:);

Cnb = 2 * [
	0.5-e2^2-e3^2 e1*e2-e0*e3 e1*e3+e0*e2
	e1*e2+e0*e3 0.5-e1^2-e3^2 e2*e3-e0*e1
	e1*e3-e0*e2 e2*e3+e0*e1 0.5-e1^2-e2^2
];

Phi_w = Ts * [
	0 w_bar(1) w_bar(2) w_bar(3)
	-w_bar(1) 0 -w_bar(3) w_bar(2)
	-w_bar(2) w_bar(3) 0 -w_bar(1)
	-w_bar(3) -w_bar(2) w_bar(1) 0
];

x_n(1:3,:) = x(1:3,:) + x(4,:) * Ts;
x_n(4:6,:) = x(4:6,:) + Cnb * a_bar * Ts;
s = 0.5 * norm(Ts * w_bar);
eta = 0.99 / Ts;
lambda = 1 - sum([ e0 ; e1 ; e2 ; e3 ].^2);
x_n(7:10,:) = (eye(4)*(cos(s) + eta*Ts*lambda) - Phi_w*0.5*sin(s)/s) * [ e0 ; e1 ; e2 ; e3 ];
x_n(11:13,:) = x(11:13,:);
x_n(14:16,:) = x(14:16,:);

if size(x,1) > 16
	x_n(11:13,:) = x_n(11:13,:) + Ts * x(27:29,:);
	x_n(14:16,:) = x_n(14:16,:) + Ts * x(30:32,:);
end
