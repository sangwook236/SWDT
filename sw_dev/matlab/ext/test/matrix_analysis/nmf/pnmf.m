function [W, H] = pnmf(X, K, maxiter, tol, threshold)

[M, N] = size(X);

W = rand(M, K);
while min(W(:)) <= 0.0
	W = rand(M, K);
end;
W = W / norm(W,2);

XXtW = zeros(M, K);
WWtXXtW = zeros(M, K);
XXtWWtW = zeros(M, K);
Wo = zeros(M, K);

for iter = 1:maxiter
	XXtW = X * (X' * W);
	WWtXXtW = W * (W' * XXtW);
	XXtWWtW = XXtW * W' * W;
	Wo = W;
	for ii = 1:M
		for jj = 1:K
			W(ii,jj) = W(ii,jj) * XXtW(ii,jj) / (WWtXXtW(ii,jj) + XXtWWtW(ii,jj));
		end;
	end;

	if norm(W - Wo, 2) < tol
		break;
	end;
end;

for ii = 1:K
	if norm(W(:,ii), 2) < threshold
		W(:,ii) = zeros(M,1);
	end;
end;
W = W / norm(W,2);

H = W' * X;
