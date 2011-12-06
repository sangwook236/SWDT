function [W, H] = ardpnmf(X, K, maxiter, tol, threshold)

[M, N] = size(X);

W = rand(M, K);
while min(W(:)) <= 0.0
	W = rand(M, K);
end;
W = W / norm(W,2);

V = zeros(1, K);
A = zeros(M, K);
B = zeros(M, K);
XXtW = zeros(M, K);
WWt = zeros(M, M);
WV = zeros(M, K);
Wo = zeros(M, K);

for iter = 1:maxiter
	for ii = 1:K
		V(ii) = 1 / (W(:,ii)' * W(:,ii));
	end;
	XXt = X * X';
	WWt = W * W';
	A = 2 * XXt * W;
	B = (WWt * XXt + XXt * WWt) * W;
	WV = W * diag(V);
	Wo = W;
	for ii = 1:M
		for jj = 1:K
			W(ii,jj) = W(ii,jj) * A(ii,jj) / (B(ii,jj) + WV(ii,jj));
		end;
	end;

	W = W / norm(W,2);

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
