function [ a, alpha, d, theta ] = calc_dh(T0i, T0j, notation)

Tij = [ T0i(1:3,1:3)' -T0i(1:3,1:3)' * T0i(1:3,4) ; 0 0 0 1 ] * T0j;

eps = 1.0e-10;

% using Craig's notation
%	link transformation -> joint transformation
if strcmpi(notation, 'craig') == true
	a = Tij(1,4);

	if abs(Tij(3,3)) < eps
		if Tij(2,3) > 0
			alpha = -pi / 2;
		else
			alpha = pi / 2;
		end;
	else
		alpha = atan2(-Tij(2,3), Tij(3,3));
	end;

	if abs(Tij(1,1)) < eps
		if Tij(1,2) > 0
			theta = -pi / 2;
		else
			theta = pi / 2;
		end;
	else
		theta = atan2(-Tij(1,2), Tij(1,1));
	end;

	sin_alpha = sin(alpha);
	cos_alpha = cos(alpha);
	if abs(sin_alpha) < eps & abs(cos_alpha) >= eps
		d = Tij(3,4) / cos_alpha;
	elseif Tij(2,4) * sin_alpha < 0
		d = sqrt(Tij(2,4)^2 + Tij(3,4)^2);
	else
		d = -sqrt(Tij(2,4)^2 + Tij(3,4)^2);
	end;
% using Paul's notation
%	joint transformation -> link transformation
elseif strcmpi(notation, 'paul') == true
	if abs(Tij(3,3)) < eps
		if Tij(3,2) > 0
			alpha = pi / 2;
		else
			alpha = -pi / 2;
		end;
	else
		alpha = atan2(Tij(3,2), Tij(3,3));
	end;

	d = Tij(3,4);

	if abs(Tij(1,1)) < eps
		if Tij(2,1) > 0
			theta = pi / 2;
		else
			theta = -pi / 2;
		end;
	else
		theta = atan2(Tij(2,1), Tij(1,1));
	end;

	sin_theta = sin(theta);
	cos_theta = cos(theta);
	if abs(sin_theta) < eps & abs(cos_theta) >= eps
		a = Tij(1,4) / cos_theta;
	elseif Tij(2,4) * sin_theta > 0
		a = sqrt(Tij(1,4)^2 + Tij(2,4)^2);
	else
		a = -sqrt(Tij(1,4)^2 + Tij(2,4)^2);
	end;
else
	error('incorrect notation');
end
