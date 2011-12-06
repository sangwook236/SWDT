function [acs, lag] = sample_acs(X, start_k, end_k)

mu = mean(X);
sigma2 = var(X, 1) * length(X);

K = end_k - start_k + 1;

lag = zeros(K, 1);
acs = zeros(K, 1);
for ii = 1:K
	lag(ii) = start_k + ii - 1;
	acs(ii) = autocorrelation(X, lag(ii), mu, sigma2);
end;
