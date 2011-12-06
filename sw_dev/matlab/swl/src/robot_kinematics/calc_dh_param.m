function [ dh_param ] = calc_dh_param(T, notation)

numLink = length(T) - 1;

dh_param = zeros(numLink, 4);

for idx = 1:numLink
	[ a, alpha, d, theta ] = calc_dh(T{idx}, T{idx+1}, notation);
	dh_param(idx,:) = [ a, alpha, d, theta ];
end;
