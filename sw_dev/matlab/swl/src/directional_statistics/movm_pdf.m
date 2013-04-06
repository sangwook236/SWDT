function prob = movm_pdf(dir, mean_dir, kappa, alpha)

% a mixture of von Mises distributions (1-dimensional)
%
% dir: a direction angle, [rad].
% mean_dir: mean direction angles, [rad].
% kappa: concentration parameters, kappa(i) >= 0.
% alpha: mixing coefficents, sum(alpha) = 1.

num1 = length(mean_dir);
%num2 = length(kappa);
%num3 = length(alpha);

%if num1 ~= num2 || num1 ~= num3
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * vm_pdf(dir, mean_dir(ii), kappa(ii));
end;
