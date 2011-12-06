function y_n = ungm_h(x_n, param)

y_n = x_n(1,:) .* x_n(1,:) ./ 20;
if size(x_n,1) == 3
	y_n = y_n + x_n(3,:);
end
