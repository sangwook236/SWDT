function prob = movmf_pdf(x, mu, kappa, alpha)

% a mixture of von Mises-Fisher distributions (n-dimensional)
%
% x: a unit direction vector on n-dimensional sphere, norm(x) = 1, column-major vector.
% mu: mean direction vectors, norm(mu(:,i)) = 1, column-major vector.
% kappa: concentration parameters, kappa(i) >= 0.
% alpha: mixing coefficents, sum(alpha) = 1.

%dim = length(x);
[ dim1 num1 ] = size(mu);
%num2 = length(kappa);
%num3 = length(alpha);

%if dim ~= dim1
%	error('dimensions are un-matched ...');
%end;
%if num1 ~= num2 || num1 ~= num3
%	error('the number of mixture components is un-matched ...');
%end;

prob = 0;
for ii = 1:num1
	prob = prob + alpha(ii) * vmf_pdf(x, mu(:,ii), kappa(ii))
end;
