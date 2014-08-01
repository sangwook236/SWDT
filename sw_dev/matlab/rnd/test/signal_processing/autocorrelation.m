function rho = autocorrelation(X, k, mu, var)

N = length(X);

XX = X - mu;

num = 0;
for ii = 1:N-k
	num = num + XX(ii + k) * XX(ii);
end;

rho = num / var;
