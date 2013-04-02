%%---------------------------------------------------------
%% create a Gaussian mixture model

mu_true = [
	1 -1
	0 2
	-2 1
];
cov_true = cat(3, [0.4 0 ; 0 0.4], [0.7 0 ; 0 0.1], [0.1 0 ; 0 0.4]);
phi_true = [ 0.5 0.25 0.25 ];

data_num = 1000;

dataset = generate_sample_from_gmm(mu_true, cov_true, phi_true, data_num);
%plot(dataset(:,1), dataset(:,2), '.')

%%---------------------------------------------------------
%% [ref] "Sparse and Shift-invariant Feature Extraction from Non-negative Data"
%% by P. Smaragdis, B. Raj, and M. Shashanka, ICASSP 2008
