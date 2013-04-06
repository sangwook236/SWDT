function prob = vmf_pdf(x, mu, kappa)

% von Mises-Fisher distributions (n-dimensional)
%
% x: a unit direction vector on n-dimensional sphere, norm(x) = 1, column-major vector.
% mu: a mean direction vector, norm(mu) = 1, column-major vector.
% kappa: a concentration parameter, kappa >= 0.

%dim = length(x);
%dim1 = length(mu);

%if dim ~= dim1
%	error('dimensions are un-matched ...');
%end;

prob = (kappa / 2)^(dim/2 - 1) * exp(kappa * dot(x, mu)) / (gamma(dim / 2) * besseli(dim/2 - 1, kappa));
