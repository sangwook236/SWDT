% "The Global Positioning System and Inertial Navigation" by Jay Farrel & Matthew Barth
% Example p. 147

% positions of the space vehicles
SVP = [
7766188.44 -21960535.34 12522838.56
-25922679.66 -6629461.28 31864.37
-5743774.02 -25828319.92 1692757.72
-2786005.69 -15900725.80 21302003.49
];

% pseudoranges
PR = [
22228206.42
24096139.11
21729070.63
21259581.09
];

% the receiver location
x = zeros(4,1);

H = ones(4,4);
rho = zeros(4,1);

XX(1,:) = x';
for kk = 1:10
	for ii = 1:4
		rho(ii) = norm(SVP(ii,:) - x(1:3)');
		H(ii,1:3) = -(SVP(ii,:) - x(1:3)') / rho(ii);
	end;

	dx = inv(H) * (PR - rho);
	x(1:3) = x(1:3) + dx(1:3);
	x(4) = dx(4);
	XX(kk+1,:) = x';
end;
