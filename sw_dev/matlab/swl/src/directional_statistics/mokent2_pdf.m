function prob = mokent2_pdf(x, kappa, beta, gamma1, gamma2, gamma3)

% a mixture of Kent distributions (2-dimensional)
%
% x: a unit direction vector on 2-dimensional sphere, norm(x) = 1, column-major vector.
% mu: mean direction vectors, norm(mu(:,i)) = 1, column-major vector.
% kappa: concentration parameters, kappa(i) >= 0.
%	The concentration of the density increases with kappa.
% beta: ellipticities of the equal probability contours of the distribution.
%	The ellipticity increases with beta(i).
%	If beta(i) = 0, the Kent distribution becomes the von Mises-Fisher distribution on the 2D sphere.
% gamma1: the mean directions.
% gamma2: the main axes of the elliptical equal probability contours.
% gamma3: the secondary axes of the elliptical equal probability contours.
%	The 3x3 matrix, [ gamma1(:,i) gamma2(:,i) gamma3(:,i) ] must be orthogonal.
%
% In the 2-dimensional case:
% The Kent distribution, also known as the 5-parameter Fisher-Bingham distribution, is a distribution on the 2D sphere (the surface of the 3D ball).
% It is the 2D member of a larger class of N-dimensional distributions called the Fisher-Bingham distributions.

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
	prob = prob + alpha(ii) * kent2_pdf(x, kappa(ii), beta(ii), gamma1(:,ii), gamma2(:,ii), gamma3(:,ii))
end;
