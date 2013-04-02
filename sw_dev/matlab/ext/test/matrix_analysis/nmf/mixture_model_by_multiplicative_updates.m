function [W, THETA] = mixture_model_by_multiplicative_updates(X, Y, BASIS_NUM, maxiter, tol)

[DATA_DIM, DATA_NUM] = size(X);
LABEL_NUM = size(Y, 2);

%initialization
W = ones(LABEL_NUM, BASIS_NUM);

THETA = zeros(DATA_DIM, BASIS_NUM);
for ii = 1:BASIS_NUM
	THETA(:,ii) = X(:, randi([1 DATA_NUM]));
end;
PHI = zeros(BASIS_NUM, DATA_NUM);
den1_n = zeros(1, DATA_NUM);
den2_n = zeros(1, DATA_NUM);
dLp_dW = zeros(LABEL_NUM, BASIS_NUM);
dLm_dW = zeros(LABEL_NUM, BASIS_NUM);
dLp_dTheta = zeros(DATA_DIM, BASIS_NUM);
dLm_dTheta = zeros(DATA_DIM, BASIS_NUM);

W0 = W;
THETA0 = THETA;

inv_eta = 1 / max(sum(X,1));
for iter = 1:maxiter
	for nn = 1:DATA_NUM
		Xn = X(:,nn);
		for jj = 1:BASIS_NUM
			%PHI(jj,nn) = exp(THETA(:,jj)' * Xn);
			PHI(jj,nn) = exp(sum(THETA(:,jj)));
		end;
	end;

	for nn = 1:DATA_NUM
		den1_n(nn) = 0;
		den2_n(nn) = 0;
		for ii = 1:LABEL_NUM
			wphi = W(ii,:) * PHI(:,nn);
			den1_n(nn) = den1_n(nn) + Y(nn,ii) * wphi;
			den2_n(nn) = den2_n(nn) + wphi;
		end;
	end;

	dLp_dW = zeros(LABEL_NUM, BASIS_NUM);
	dLm_dW = zeros(LABEL_NUM, BASIS_NUM);
	for kk = 1:LABEL_NUM
		for mm = 1:BASIS_NUM
			for nn = 1:DATA_NUM
				dLp_dW(kk,mm) = dLp_dW(kk,mm) + Y(nn,kk) * PHI(mm,nn) / den1_n(nn);
				dLm_dW(kk,mm) = dLm_dW(kk,mm) + PHI(mm,nn) / den2_n(nn);
			end;
		end;
	end;

	W0 = W;
	for kk = 1:LABEL_NUM
		for mm = 1:BASIS_NUM
			W(kk,mm) = W(kk,mm) * dLp_dW(kk,mm) / dLm_dW(kk,mm);
		end;
	end;

	dLp_dTheta = zeros(DATA_DIM, BASIS_NUM);
	dLm_dTheta = zeros(DATA_DIM, BASIS_NUM);
	for mm = 1:BASIS_NUM
		for uu = 1:DATA_DIM
			for nn = 1:DATA_NUM
				dLp_dTheta(uu,mm) = dLp_dTheta(uu,mm) + (Y(nn,:) * W(:,mm)) * PHI(mm,nn) * X(uu,nn) / den1_n(nn);
				dLm_dTheta(uu,mm) = dLm_dTheta(uu,mm) + sum(W(:,mm)) * PHI(mm,nn) * X(uu,nn) / den2_n(nn);
			end;
		end;
	end;

	THETA0 = THETA;
	for mm = 1:BASIS_NUM
		for uu = 1:DATA_DIM
			THETA(uu,mm) = log(exp(THETA(uu,mm)) * (dLp_dTheta(uu,mm) / dLm_dTheta(uu,mm))^(inv_eta));
		end;
	end;

	if norm(W - W0, 2) < tol && norm(THETA - THETA0, 2) < tol
		break;
	end;

	iter
	W
	%THETA
	PHI
	%den1_n
	%den2_n
end;
