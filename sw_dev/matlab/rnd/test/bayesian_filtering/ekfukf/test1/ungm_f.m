function x_n = ungm_f(x, param)

n = param(1);
x_n = 0.5*x(1,:) + 25*x(1,:)./(1+x(1,:).*x(1,:)) + 8*cos(1.2*(n-1));
if size(x,1) > 1
	x_n = x_n + x(2,:);
end;
